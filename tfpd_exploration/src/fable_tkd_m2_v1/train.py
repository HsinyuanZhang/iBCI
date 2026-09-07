"""Training for FABLE TKD M2 arms (Wave 2, spec sections 0/2).

GPU1 (or CPU for smoke), float32, TF32 off, fail-closed.  Loss is raw-target
MSE (targets are UNSCALED covariates; the model already divides its output by
BEHAVIOR_SCALE internally) -- this differs from the champion's x5.0-scaled
loss only by a global loss scale and is recorded in every train receipt.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as screen_core

from . import data, plan
from .model import TKD


def load_session_plane(repo_root: Path) -> dict[str, Any]:
    """Frozen datasets + closed-form per-session (t, rho) with the sealed T4
    authority verified bit-for-bit (Wave 1 stage0 receipt)."""
    bundle = data.load_frozen_bundle()
    fits: dict[str, data.SessionFit] = {}
    for surface_key, sessions in (
        ("within", plan.HELDIN_SESSIONS),
        ("external", plan.EXTERNAL_SESSIONS),
    ):
        for session in sessions:
            fits[session] = data.m30_t4(bundle[surface_key], session)
    authority = data.build_or_verify_authority(
        fits, plan.result_root(repo_root) / "stage0"
    )
    mean = np.asarray(authority["mean"], dtype=np.float32)
    std = np.asarray(authority["std"], dtype=np.float32)
    for fit in fits.values():
        fit.t = data.normalize_t4(fit.raw, mean, std)
    min_t = min(float(fit.t.min()) for fit in fits.values())
    plan.require(min_t > -45.0, f"normalized T4 violates Phi_k bypass region ({min_t})")
    return {"bundle": bundle, "fits": fits, "authority": authority}


def ensure_teacher_cache(
    plane: dict[str, Any], device: torch.device, repo_root: Path
) -> dict[str, np.ndarray]:
    """D14 teacher targets: frozen champion last-timestep x5-space outputs on
    the held-in post-30 training windows, cached once per session (npy +
    sha256 sidecar, 0444).  Teacher identity follows the champion's own
    deployment law: first-30 calibration block + ridge m30 T4 side features
    with the champion's frozen normalizer (probe/screen machinery)."""
    repo_root = Path(repo_root)
    cache_dir = repo_root / plan.TEACHER_CACHE_RELATIVE
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle = plane["bundle"]
    dataset = bundle["within"]
    cached: dict[str, np.ndarray] = {}
    rebuilt: list[str] = []
    for session in plan.HELDIN_SESSIONS:
        path = cache_dir / f"{session}.npy"
        if path.exists():
            sidecar = path.with_name(path.name + ".sha256")
            plan.require(sidecar.is_file(), f"teacher cache sidecar missing: {session}")
            array = np.load(path)
            digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
            plan.require(
                sidecar.read_text(encoding="utf-8") == f"{digest}  {path.name}\n",
                f"teacher cache digest drift: {session}",
            )
            cached[session] = array
            continue
        model = bundle.get("champion_model")
        plan.require(model is not None,
                     "champion model missing from bundle; teacher cache rebuild impossible")
        student = model.student.to(device).eval()
        from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as screen_physical

        selected = np.arange(plan.ACTIVITY_HORIZON, dtype=np.int64)
        side, _evidence = screen_physical._ridge_side(dataset, session, selected)
        calibration = np.asarray(
            dataset.calib_trialized_neural_features[session], dtype=np.float32
        )
        activity = np.ascontiguousarray(
            calibration[: plan.ACTIVITY_HORIZON], dtype=np.float32
        )
        starts = post30_starts(dataset, session)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            support = torch.from_numpy(activity).unsqueeze(0).to(device)
            side_tensor = torch.from_numpy(side).unsqueeze(0).to(device)
            identity = student.compute_identity(support, side_features=side_tensor)
            for offset in range(0, starts.size, plan.EVAL_BATCH):
                chunk = starts[offset : offset + plan.EVAL_BATCH]
                neural_np = np.stack(
                    [dataset.neural_data[session][start : start + plan.WINDOW]
                     for start in chunk], axis=0
                ).astype(np.float32, copy=False)
                neural = torch.from_numpy(neural_np).to(device)
                prediction, _ = student(neural, identity=identity)
                outputs.append(
                    prediction[:, -1, :].detach().cpu().numpy().astype(np.float32)
                )
        targets = np.ascontiguousarray(np.concatenate(outputs, axis=0))
        plan.require(targets.shape == (starts.size, plan.OUT_DIM)
                     and np.isfinite(targets).all(), "teacher target drift")
        temporary = path.with_name(f".{path.name}.npy.tmp")
        with open(temporary, "wb") as handle:
            np.save(handle, targets)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
        # sidecar digest covers the ARRAY bytes (dtype/shape-bound via the
        # receipt's array_sha256); re-verified on every load.
        digest = hashlib.sha256(targets.tobytes(order="C")).hexdigest()
        sidecar = path.with_name(path.name + ".sha256")
        sidecar_tmp = path.with_name(f".{path.name}.sha256.tmp")
        with open(sidecar_tmp, "wb") as handle:
            handle.write(f"{digest}  {path.name}\n".encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_tmp.replace(sidecar)
        sidecar.chmod(0o444)
        reloaded = np.load(path)
        plan.require(
            hashlib.sha256(reloaded.tobytes(order="C")).hexdigest() == digest,
            "teacher cache reload digest drift",
        )
        cached[session] = reloaded
        rebuilt.append(session)
    if rebuilt:
        receipt = {
            "schema": f"{plan.SCHEMA}:teacher_cache",
            "sessions": {
                s: {
                    "n_windows": int(cached[s].shape[0]),
                    "sha256_bytes": hashlib.sha256(cached[s].tobytes(order="C")).hexdigest(),
                    "array_sha256": screen_core.array_sha256(cached[s]),
                }
                for s in plan.HELDIN_SESSIONS
            },
            "rebuilt": rebuilt,
            "teacher_checkpoint_sha256": plan.CHAMPION_CKPT_SHA256,
            "law": (
                "frozen champion student, identity = compute_identity(first-30 "
                "activity, ridge m30 T4 side with champion frozen normalizer); "
                "targets = last-timestep outputs (x5 space)"
            ),
        }
        plan.atomic_receipt(cache_dir / "teacher_cache.json", receipt)
    return cached


def post30_starts(dataset: Any, session: str) -> np.ndarray:
    starts = data.session_window_starts(dataset, session)
    return screen_core.select_common_post30_window_starts(
        starts, dataset.trial_start_indices[session]
    )


def gather_windows(
    dataset: Any, session: str, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unscaled) windows and targets for the given starts."""
    return data._stack_windows(dataset, session, starts, scale_targets=False)


def _tensor_sha(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return screen_core.array_sha256(array)


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Per-tensor sha over sorted keys (the probe's _state_sha law)."""
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(_tensor_sha(state[key]).encode())
    return digest.hexdigest()


def _apply_seed_law(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_tf32_off() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan.require(torch.backends.cuda.matmul.allow_tf32 is False, "TF32 matmul not off")
    plan.require(torch.backends.cudnn.allow_tf32 is False, "TF32 cudnn not off")


def _build_arm_model(arm: str, plane: dict[str, Any], device: torch.device) -> TKD:
    authority = plane["authority"]
    model = TKD(
        **plan.ARM_TABLE[arm],
        authority_mean=np.asarray(authority["mean"], dtype=np.float32),
        authority_std=np.asarray(authority["std"], dtype=np.float32),
        pv_beta=float(plan.PV_INIT_CONFIG["pv_beta"]),
        pv_bin_masses=np.asarray(plan.PV_INIT_CONFIG["bin_masses"], dtype=np.float32),
        pv_output_scale=plan.PV_INIT_OUTPUT_SCALE,
        pv_readout_bias=plan.PV_READOUT_BIAS_X5,
    )
    return model.to(device)


def _score_starts(
    model: TKD,
    dataset: Any,
    sessions: tuple[str, ...],
    starts_for: Any,
    fits: dict[str, Any],
    device: torch.device,
    *,
    batch_size: int = plan.EVAL_BATCH,
) -> dict[str, Any]:
    """Per-session variance-weighted R2 over caller-provided starts."""
    model.eval()
    per_session: dict[str, float] = {}
    with torch.no_grad():
        for session in sessions:
            starts = np.asarray(starts_for(session))
            if starts.size == 0:
                continue
            fit = fits[session]
            tensor_t = torch.from_numpy(np.ascontiguousarray(fit.t)).to(device)
            tensor_rho = torch.from_numpy(np.ascontiguousarray(fit.rho)).to(device)
            predictions: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for offset in range(0, starts.size, batch_size):
                chunk = starts[offset : offset + batch_size]
                windows, window_targets = gather_windows(dataset, session, chunk)
                prediction = model(
                    torch.from_numpy(windows).to(device), tensor_t, tensor_rho
                )
                predictions.append(prediction.cpu().numpy())
                targets.append(window_targets)
            prediction_np = np.ascontiguousarray(np.concatenate(predictions, axis=0))
            target_np = np.ascontiguousarray(np.concatenate(targets, axis=0))
            plan.require(prediction_np.shape == target_np.shape, "eval shape drift")
            per_session[session] = screen_core.variance_weighted_r2(
                target_np, prediction_np
            )
    summary = screen_core.summarize_sessions(per_session)
    return {"per_session_r2": per_session, **summary}


def evaluate_within(
    model: TKD,
    plane: dict[str, Any],
    device: torch.device,
    *,
    batch_size: int = plan.EVAL_BATCH,
) -> dict[str, Any]:
    """Within post-30 face (ALL post-30 windows; unchanged eval face)."""
    bundle, fits = plane["bundle"], plane["fits"]
    dataset = bundle["within"]
    return _score_starts(
        model, dataset, plan.HELDIN_SESSIONS,
        lambda session: post30_starts(dataset, session),
        fits, device, batch_size=batch_size,
    )


def evaluate_external(
    model: TKD,
    plane: dict[str, Any],
    device: torch.device,
    *,
    batch_size: int = plan.EVAL_BATCH,
) -> dict[str, Any]:
    """External official face (all windows; unchanged eval face)."""
    bundle, fits = plane["bundle"], plane["fits"]
    dataset = bundle["external"]
    return _score_starts(
        model, dataset, plan.EXTERNAL_SESSIONS,
        lambda session: data.session_window_starts(dataset, session),
        fits, device, batch_size=batch_size,
    )


def train_one(
    repo_root: Path,
    arm: str,
    seed: int,
    device: torch.device,
    *,
    plane: dict[str, Any] | None = None,
    teacher_weight: float = 0.0,
    run_tag: str = "",
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(repo_root)
    plan.require(arm in plan.ARM_TABLE, f"unknown arm {arm}")
    plane = plane if plane is not None else load_session_plane(repo_root)
    _apply_seed_law(seed)
    _assert_tf32_off()
    model = _build_arm_model(arm, plane, device)
    bundle, fits, authority = plane["bundle"], plane["fits"], plane["authority"]
    dataset = bundle["within"]

    # Session-pure batches: t/rho are per-session model inputs, so the global
    # window shuffle is realized as (session order, within-session window
    # order) permutations from one np.random.default_rng(seed) stream.
    rng = np.random.default_rng(seed)
    sessions = list(plan.HELDIN_SESSIONS)
    # D16: chronological per-session 80/20 split of the post-30 windows --
    # first 80% train, last 20% minival (disjoint selection face).
    train_starts: dict[str, np.ndarray] = {}
    minival_starts: dict[str, np.ndarray] = {}
    all_starts = {s: post30_starts(dataset, s) for s in sessions}
    for session in sessions:
        starts = all_starts[session]
        n_train = int(round(starts.size * (1.0 - plan.MINIVAL_FRACTION)))
        plan.require(n_train >= 1 and starts.size - n_train >= 1,
                     f"degenerate split for {session}")
        train_starts[session] = np.ascontiguousarray(starts[:n_train])
        minival_starts[session] = np.ascontiguousarray(starts[n_train:])
    total_windows = int(sum(train_starts[s].size for s in sessions))

    tensors = {}
    for session in sessions:
        fit = fits[session]
        tensors[session] = (
            torch.from_numpy(np.ascontiguousarray(fit.t)).to(device),
            torch.from_numpy(np.ascontiguousarray(fit.rho)).to(device),
        )

    # D14 teacher targets (held-in only; cached once, digest-sealed).  D16:
    # only the TRAIN split rows are consumed; minival teacher rows are never
    # touched by training.
    teacher: dict[str, np.ndarray] = {}
    teacher_digests: dict[str, str] = {}
    if teacher_weight > 0.0:
        teacher = ensure_teacher_cache(plane, device, repo_root)
        for session in sessions:
            plan.require(teacher[session].shape[0] == all_starts[session].size,
                         f"teacher cache window-count drift for {session}")
            n_train = train_starts[session].size
            teacher_digests[session] = hashlib.sha256(
                teacher[session][:n_train].tobytes(order="C")
            ).hexdigest()
            tensors[session] = tensors[session] + (
                torch.from_numpy(np.ascontiguousarray(teacher[session][:n_train])).to(device),
            )
    else:
        for session in sessions:
            tensors[session] = tensors[session] + (None,)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=plan.LR, weight_decay=plan.WD)
    history: list[dict[str, Any]] = []
    best_mean = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(plan.EPOCHS):
        model.train()
        order = rng.permutation(len(sessions))
        losses: list[float] = []
        for session_index in order:
            session = sessions[int(session_index)]
            starts = train_starts[session]
            window_order = rng.permutation(starts.size)
            tensor_t, tensor_rho, tensor_teacher = tensors[session]
            for offset in range(0, window_order.size, plan.BATCH):
                chosen = window_order[offset : offset + plan.BATCH]
                selected = starts[chosen]
                windows, targets = gather_windows(dataset, session, selected)
                x = torch.from_numpy(windows).to(device)
                y = torch.from_numpy(targets).to(device)
                prediction = model(x, tensor_t, tensor_rho)
                # D13(a): loss on x5-scaled targets (champion convention):
                # readout_out = prediction * 5 vs y_raw * 5; the /5 division
                # stays at eval/deployment only.  D14: + lambda * MSE vs the
                # frozen champion teacher's last-timestep outputs (x5 space).
                readout_out = prediction * plan.BEHAVIOR_SCALE
                loss = torch.nn.functional.mse_loss(
                    readout_out, y * plan.BEHAVIOR_SCALE
                )
                if tensor_teacher is not None:
                    loss = loss + teacher_weight * torch.nn.functional.mse_loss(
                        readout_out, tensor_teacher[chosen]
                    )
                plan.require(bool(torch.isfinite(loss).item()),
                             f"non-finite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, plan.GRAD_CLIP)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        # D16 selection metric: disjoint minival equal-session mean.  Within
        # (full post-30) and external are recorded per epoch as DISCLOSED
        # DIAGNOSTICS only (external-vs-epoch is for the continuation rule,
        # never for epoch choice).
        minival = _score_starts(
            model, dataset, tuple(sessions),
            lambda s: minival_starts[s], fits, device,
        )
        within = evaluate_within(model, plane, device)
        external = evaluate_external(model, plane, device)
        mean = float(minival["equal_session_mean"])
        history.append(
            {
                "epoch": epoch,
                "mean_train_mse_raw_target": float(np.mean(losses)),
                "batch_count": len(losses),
                "minival_equal_session_mean": mean,
                "minival_per_session_r2": minival["per_session_r2"],
                "within_equal_session_mean":
                    float(within["equal_session_mean"]),
                "external_equal_session_mean":
                    float(external["equal_session_mean"]),
            }
        )
        if mean > best_mean:  # strict: ties keep the earlier epoch
            best_mean = mean
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"[{arm} s{seed}] epoch {epoch:02d} mse={float(np.mean(losses)):.6f} "
            f"minival={mean:.6f} within={within['equal_session_mean']:.6f} "
            f"external={external['equal_session_mean']:.6f}",
            flush=True,
        )
    plan.require(best_state is not None, "no best epoch selected")

    run_dir = (
        plan.result_root(repo_root) / "runs" / f"{arm}_s{seed}{run_tag}"
        if run_tag
        else plan.result_root(repo_root) / "runs" / f"{arm}_s{seed}"
    )
    if run_dir.exists():
        # Retry after a failed attempt: sealed artifacts stay in place; the
        # new attempt uses an explicit _a<n> suffix recorded in its receipt.
        index = 2
        while (run_dir.parent / f"{run_dir.name}_a{index}").exists():
            index += 1
        run_dir = run_dir.parent / f"{run_dir.name}_a{index}"
    run_dir.mkdir(parents=True, exist_ok=False)
    buffer = __import__("io").BytesIO()
    torch.save(best_state, buffer)
    best_bytes = buffer.getvalue()
    fd_path = run_dir / "best.pt"
    _atomic_bytes(fd_path, best_bytes)
    state_sha = state_dict_sha256(best_state)

    receipt = {
        "schema": f"{plan.SCHEMA}:train",
        "arm": arm,
        "arm_config": plan.ARM_TABLE[arm],
        "seed": seed,
        "pv_init_config": plan.PV_INIT_CONFIG,
        "constants": {
            "epochs": plan.EPOCHS, "lr": plan.LR, "weight_decay": plan.WD,
            "batch": plan.BATCH, "grad_clip": plan.GRAD_CLIP,
            "eval_batch": plan.EVAL_BATCH,
            "loss": "mse_on_x5_scaled_targets",
            "loss_note": (
                "D13(a)/ADDENDUM-4: loss = MSE(readout_out, y_raw*5.0) -- the "
                "champion behavior_scaling_factor convention; the /5.0 "
                "division stays at eval/deployment only (supersedes the "
                "Wave 2 raw-target loss spec choice)"
            ),
            "pv_output_scale": plan.PV_INIT_OUTPUT_SCALE,
            "pv_readout_bias_x5": list(plan.PV_READOUT_BIAS_X5),
            "teacher_weight": float(teacher_weight),
            "teacher_law": (
                "L = MSE(readout_out, y*5) + teacher_weight * MSE(readout_out, "
                "teacher_out); teacher = frozen champion (m30 identity law), "
                "last-timestep x5 outputs, held-in windows only" if teacher_weight > 0
                else "none (from-scratch loss)"
            ),
            "ssm_state_bounding": "a = 0.98*sigmoid(a_log_raw), h clamped to +/-10, gamma = tanh(gamma_raw)",
            "shuffle_law": (
                "one np.random.default_rng(seed) stream; per epoch draw a "
                "session-order permutation then a within-session window-order "
                "permutation (batches are session-pure because t/rho are "
                "per-session model inputs)"
            ),
        },
        "total_trainable_parameters": int(sum(p.numel() for p in trainable)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "frozen_readin": bool(plan.ARM_TABLE[arm]["frozen_readin"]),
        "window_total": total_windows,
        "windows_per_session": {s: int(train_starts[s].size) for s in sessions},
        "minival_windows_per_session": {
            s: int(minival_starts[s].size) for s in sessions
        },
        "selection_law": (
            "D16: chronological per-session 80/20 post-30 split; deployment "
            "epoch = argmax of the disjoint-minival equal-session mean "
            "(tie -> earlier); within/external per-epoch values are disclosed "
            "diagnostics only"
        ),
        "run_dir": str(run_dir),
        "tf32_matmul_off": True,
        "tf32_cudnn_off": True,
        "gpu_uuid": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "device": str(device),
        "authority_sha256": plan.sha256_bytes(plan.receipt_body(authority)),
        "t4_digests": {
            s: {
                "raw_sha256": fits[s].evidence["raw_t4_sha256"],
                "rho_sha256": fits[s].evidence["rho_sha256"],
            }
            for s in plan.ALL_SESSIONS
        },
        "teacher_target_sha256_bytes": teacher_digests,
        "history": history,
        "selected_epoch": int(best_epoch),
        "selected_minival_equal_session_mean": best_mean,
        "best_state_sha256": state_sha,
        "best_pt_sha256": hashlib.sha256(best_bytes).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(run_dir / "train.json", receipt)
    return receipt


def _atomic_bytes(path: Path, body: bytes) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise plan.TKDError(f"refusing to overwrite {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    sidecar_temporary = temporary.with_name(temporary.name + ".sha256")
    sidecar = path.with_name(path.name + ".sha256")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(body).hexdigest()
        temporary.replace(path)
        path.chmod(0o444)
        with open(sidecar_temporary, "wb") as handle:
            handle.write(f"{digest}  {path.name}\n".encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_temporary.replace(sidecar)
        sidecar.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
