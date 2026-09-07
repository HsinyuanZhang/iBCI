"""Training + evaluation for FABLE TKD M1 v1 (fold-local pilot).

Conventions mirrored verbatim from the fold-local champion (M1-D1): Adam
lr 1e-4 / wd 0, batch 32, 12 epochs, FIXED LAST EPOCH (no selection), raw
16-dim EMG targets (behavior_scaling_factor = 1).  Loss adds the pinned
lambda_E = 1.0 distillation term against the frozen SPINT teacher's cached
outputs (source training windows only).
"""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m1_heldin_heldout_gap_v1.physical import (
    variance_weighted_last_bin_r2,
)

from tfpd_exploration.src.fable_tkd_m2_v1.model import TKD

from . import data as m1_data
from . import plan
from .anchor import build_anchor

ARM_TABLE = {
    "A_PRIME": dict(epsilon="learnable", key_mode="identity"),
    "SHUF": dict(epsilon="learnable", key_mode="shuffled"),
}


def build_arm_model(arm: str, plane: dict[str, Any], device: torch.device) -> TKD:
    plan.require(arm in ARM_TABLE, f"unknown arm {arm}")
    torch.manual_seed(plan.SEED)
    model = TKD(
        time_model="gru",
        init="nmf",
        frozen_readin=False,
        channels=plan.CHANNELS,
        window=plan.WINDOW_SIZE,
        out_dim=plan.OUT_DIM,
        behavior_scale=plan.BEHAVIOR_SCALE,
        shuf_perm=plan.SHUF_PERM,
        authority_mean=np.asarray(plane["bank"]["normalizer_mean"], dtype=np.float32),
        authority_std=np.asarray(plane["bank"]["normalizer_scale"], dtype=np.float32),
        nmf_anchor=build_anchor(plane),
        **ARM_TABLE[arm],
    )
    return model.to(device)


def _apply_seed_law(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gather(base: Any, session: str, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neural = np.asarray(base.neural_data[session], dtype=np.float32)
    covariate = np.asarray(base.covariate_data[session], dtype=np.float32)
    windows = np.stack([neural[s : s + plan.WINDOW_SIZE] for s in starts])
    targets = covariate[starts + plan.WINDOW_SIZE - 1]
    return (np.ascontiguousarray(windows), np.ascontiguousarray(targets))


def evaluate_face(model: TKD, plane: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Fold-0 target query face (verbatim): last-bin R2 via the sealed metric."""
    model.eval()
    base = plane["eval_base"]
    session = plan.FOLD0_TARGET_SESSION
    starts = np.asarray(
        [s for name, s in base.window_indices if name == session], dtype=np.int64
    )
    plan.require(starts.size == plan.EXPECTED_EVAL_WINDOWS, "eval window drift")
    t, rho = m1_data.session_tensors(plane, session)
    tensor_t = torch.from_numpy(np.ascontiguousarray(t)).to(device)
    tensor_rho = torch.from_numpy(np.ascontiguousarray(rho)).to(device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, starts.size, 256):
            chunk = starts[offset : offset + 256]
            windows, window_targets = _gather(base, session, chunk)
            prediction = model(
                torch.from_numpy(windows).to(device), tensor_t, tensor_rho
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(window_targets)
    prediction_np = np.ascontiguousarray(np.concatenate(predictions, axis=0))
    target_np = np.ascontiguousarray(np.concatenate(targets, axis=0))
    return {
        "governing_r2": variance_weighted_last_bin_r2(prediction_np, target_np),
        "n_windows": int(starts.size),
        "prediction_sha256": m1_data._digest_array(prediction_np),
        "target_sha256": m1_data._digest_array(target_np),
    }


def train_arm(
    repo_root: Path,
    arm: str,
    device: torch.device,
    plane: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(repo_root)
    plane = plane if plane is not None else m1_data.load_plane(repo_root)
    _apply_seed_law(plan.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_arm_model(arm, plane, device)
    base = plane["train_base"]
    teacher = m1_data.ensure_teacher_cache(plane, repo_root, device)

    # Static session-pure schedule drawn ONCE (the fold-local champion's
    # reshuffle_train_sampler_each_epoch=false convention), reused every epoch.
    rng = np.random.default_rng(plan.SEED)
    sessions = list(plan.FOLD0_SOURCE_SESSIONS)
    tensors = {}
    for session in sessions:
        t, rho = m1_data.session_tensors(plane, session)
        tensors[session] = (
            torch.from_numpy(np.ascontiguousarray(t)).to(device),
            torch.from_numpy(np.ascontiguousarray(rho)).to(device),
            torch.from_numpy(
                np.ascontiguousarray(teacher[session])
            ).to(device),
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=plan.LR, weight_decay=plan.WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    for epoch in range(plan.EPOCHS):
        model.train()
        order = rng.permutation(len(sessions))
        losses: list[float] = []
        distill_losses: list[float] = []
        for session_index in order:
            session = sessions[int(session_index)]
            starts = plane["train_starts"][session]
            window_order = rng.permutation(starts.size)
            tensor_t, tensor_rho, tensor_teacher = tensors[session]
            for offset in range(0, window_order.size, plan.BATCH):
                chosen = window_order[offset : offset + plan.BATCH]
                selected = starts[chosen]
                windows, targets = _gather(base, session, selected)
                x = torch.from_numpy(windows).to(device)
                y = torch.from_numpy(targets).to(device)
                prediction = model(x, tensor_t, tensor_rho)
                task = torch.nn.functional.mse_loss(prediction, y)
                distill = torch.nn.functional.mse_loss(
                    prediction, tensor_teacher[chosen]
                )
                loss = task + plan.DISTILL_LAMBDA * distill
                plan.require(bool(torch.isfinite(loss).item()),
                             f"non-finite loss epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, plan.GRAD_CLIP)
                optimizer.step()
                losses.append(float(task.detach().cpu()))
                distill_losses.append(float(distill.detach().cpu()))
        face = evaluate_face(model, plane, device)  # disclosed diagnostic only
        history.append({
            "epoch": epoch,
            "task_mse": float(np.mean(losses)),
            "distill_mse": float(np.mean(distill_losses)),
            "face_r2_diagnostic": face["governing_r2"],
        })
        print(
            f"[{arm} s{plan.SEED}] epoch {epoch:02d} task={np.mean(losses):.6f} "
            f"distill={np.mean(distill_losses):.6f} face_r2={face['governing_r2']:.4f}",
            flush=True,
        )
    plan.require(plan.EPOCHS - 1 == plan.FIXED_LAST_EPOCH_INDEX, "fixed-last drift")

    run_dir = plan.result_root(repo_root) / "runs" / f"{arm}_s{plan.SEED}"
    if run_dir.exists():
        index = 2
        while (run_dir.parent / f"{run_dir.name}_a{index}").exists():
            index += 1
        run_dir = run_dir.parent / f"{run_dir.name}_a{index}"
    run_dir.mkdir(parents=True, exist_ok=False)
    import io

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    best_bytes = buffer.getvalue()
    import os
    import tempfile

    fd, tmp_name = tempfile.mkstemp(prefix=".best.pt.", dir=run_dir)
    with os.fdopen(fd, "wb") as handle:
        handle.write(best_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_name, run_dir / "best.pt")
    (run_dir / "best.pt").chmod(0o444)
    digest = hashlib.sha256(best_bytes).hexdigest()
    sidecar = run_dir / "best.pt.sha256"
    sidecar.write_text(f"{digest}  best.pt\n", encoding="utf-8")
    sidecar.chmod(0o444)

    face = evaluate_face(model, plane, device)
    receipt = {
        "schema": f"{plan.SCHEMA}:train",
        "arm": arm,
        "arm_config": {**ARM_TABLE[arm], "time_model": "gru", "init": "nmf"},
        "seed": plan.SEED,
        "constants": {
            "epochs": plan.EPOCHS, "fixed_last_epoch_index": plan.FIXED_LAST_EPOCH_INDEX,
            "lr": plan.LR, "weight_decay": plan.WEIGHT_DECAY, "batch": plan.BATCH,
            "grad_clip": plan.GRAD_CLIP, "distill_lambda": plan.DISTILL_LAMBDA,
            "behavior_scale": plan.BEHAVIOR_SCALE,
            "targets": "raw 16-dim EMG last bin (predict_scaled_behavior=false law)",
            "shuffle_law": "one np.random.default_rng(42) stream; static "
                           "session-pure schedule (champion reshuffle=false law)",
            "note_grad_clip": "grad clip 1.0 retained from the M2 pipeline; "
                              "the fold-local champion Lightning trainer used none",
        },
        "fold0_target_session": plan.FOLD0_TARGET_SESSION,
        "source_sessions": list(plan.FOLD0_SOURCE_SESSIONS),
        "sigma_pooled": plane["sigma_pooled"],
        "anchor_beta": plan.ANCHOR_BETA,
        "teacher_sha256_bytes": {
            s: hashlib.sha256(teacher[s].tobytes(order="C")).hexdigest()
            for s in plan.FOLD0_SOURCE_SESSIONS
        },
        "history": history,
        "selected_epoch": plan.FIXED_LAST_EPOCH_INDEX,
        "face_at_fixed_last": face,
        "best_pt_sha256": digest,
        "run_dir": str(run_dir),
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(run_dir / "train.json", receipt)
    return receipt
