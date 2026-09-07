"""Operational frozen M2 champion for m2_dual_track_v1.

Rebuilds MOVE-T4 + trained EMPTY (profile-free) adapter WITHOUT calling
``load_frozen_model_and_data`` and WITHOUT ``data_module.setup("test")``.

Output-space contract
---------------------
``Candidate.forward_last`` returns **decoder_raw**: the frozen student's
last-bin output.  The Evaluator divides by ``plan.BEHAVIOR_SCALE`` (5.0)
before variance-weighted R2 against **native** finger-velocity covariates.
See ``tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py``.
Do not divide here.  Do not skip the divide at score time.

Profile-free / EMPTY means the contrast *input* is zeros, not that adapter
weights are zero.  The trained head still consumes ``[MOVE-T4, 0]`` and
applies ``(1+gamma)h+beta`` on early-pooled features.

E0 is produced only by the native chronological FP32 path
(``push_trial`` / ``finalize_identity``).  Do not replace it with
``torch.mean`` of a batched GEMM.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import contracts, plan

OUTPUT_SPACE_CONTRACT = {
    "forward_last": plan.TRAINING_TARGET_SPACE,
    "scoring": plan.SCORING_TARGET_SPACE,
    "behavior_scale": plan.BEHAVIOR_SCALE,
    "divide_at": "evaluator_only",
    "reference": (
        "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py "
        "divides decoder_raw by BEHAVIOR_SCALE=5 before R2 vs native covariates"
    ),
}

FILM_RANK = 8
CONTRAST_DIM = 4
SIDE_DIM = plan.T4_DIM + CONTRAST_DIM
TEACHER_CKPT_RELATIVE = (
    "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"
)
TEACHER_CKPT_SHA256 = "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"


def _streaming_root() -> Path:
    return plan.repo_root() / "streaming_calibration_exp"


def ensure_streaming_on_path() -> Path:
    root = _streaming_root()
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        tensor = state[key]
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        header = json.dumps(
            {"dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        leaf = hashlib.sha256()
        leaf.update(header)
        leaf.update(array.tobytes(order="C"))
        digest.update(leaf.hexdigest().encode())
    return digest.hexdigest()


def _post_linear_count(encoder: nn.Module) -> int:
    return sum(1 for child in encoder.post_pool if isinstance(child, nn.Linear))


def cache_key_parts() -> dict[str, Any]:
    return {
        "ckpt_sha256": plan.CHAMPION_CKPT_SHA256,
        "head_state_sha256": plan.SELECTED_HEAD_STATE_SHA256,
        "film_states": plan.FILM_STATES_RELATIVE,
        "canonical_encoder": "p0",
        "preprocessing": "move_t4_bins[5,30)_empty_contrast_fp32",
        "dtype": "fp32",
        "support_horizon": plan.SUPPORT_HORIZON,
        "window": plan.WINDOW,
        "channels": plan.CHANNELS,
        "output_space": OUTPUT_SPACE_CONTRACT,
        "contract_version": plan.CONTRACT_VERSION,
    }


def load_base_lightning_module(*, device: torch.device) -> Any:
    """Construct the pinned student from the champion ckpt. Source-only: no data.

    Uses ``model.setup("fit")`` only to build teacher→student weights.  This is
    not ``data_module.setup("test")`` and does not open NWBs.
    """
    ensure_streaming_on_path()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    ckpt = plan.repo_root() / plan.CHAMPION_CKPT_RELATIVE
    config_path = plan.repo_root() / plan.CHAMPION_RUN_RELATIVE / "resolved_config.yaml"
    teacher = plan.repo_root() / TEACHER_CKPT_RELATIVE
    plan.require(ckpt.is_file(), f"missing champion ckpt: {ckpt}")
    plan.require(config_path.is_file(), f"missing resolved config: {config_path}")
    plan.require(teacher.is_file(), f"missing teacher ckpt: {teacher}")
    ckpt_sha = sha256_file(ckpt)
    plan.require(ckpt_sha == plan.CHAMPION_CKPT_SHA256, f"champion ckpt drift: {ckpt_sha}")
    teacher_sha = sha256_file(teacher)
    plan.require(teacher_sha == TEACHER_CKPT_SHA256, f"teacher ckpt drift: {teacher_sha}")

    config = OmegaConf.load(config_path)
    # Do not OmegaConf.resolve the full file: it still contains ${hydra:...}.
    config.model.teacher_ckpt_path = str(teacher)
    config.model.window_size = int(config.data.window_size)
    config.model.trial_length = int(config.data.max_trial_length)
    config.model.pad_value = float(config.data.pad_value)
    model = instantiate(config.model)
    model.setup("fit")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    plan.require(model.student is not None, "student missing after setup")
    plan.require(model.student.decoder_mode == "coupled", "decoder_mode drift")
    plan.require(bool(model.student._decoder_frozen), "decoder not frozen on load")
    model.teacher = None
    return model


def install_empty_film(student: Any, device: torch.device) -> Any:
    """Replace B3S with HoldContrastFiLM. Does not import install_film/score_arm."""
    from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import (
        HoldContrastFiLMEarlyPoolEncoder,
    )

    base = student.id_encoder
    plan.require(getattr(base, "variant", None) == "B3S", "frozen encoder is not B3S")
    plan.require(int(base.side_dim) == plan.T4_DIM, "frozen B3S side_dim is not 4")
    plan.require(int(base.hidden_dim) == plan.HIDDEN_DIM, "hidden_dim drift")
    plan.require(int(base.trial_length) == plan.CALIB_TRIAL_LENGTH, "trial_length drift")
    plan.require(int(base.window_size) == plan.WINDOW, "window_size drift")
    in_features = int(base.post_pool[0].in_features)
    plan.require(in_features == plan.HIDDEN_DIM + plan.T4_DIM, "post_pool geometry is not 64+4")

    film = HoldContrastFiLMEarlyPoolEncoder(
        trial_length=int(base.trial_length),
        window_size=int(base.window_size),
        hidden_dim=int(base.hidden_dim),
        side_dim=SIDE_DIM,
        film_rank=FILM_RANK,
        num_post_layers=_post_linear_count(base),
        film_input="t4_plus_contrast",
    )
    film.load_t4_state_dict({key: value.detach().cpu() for key, value in base.state_dict().items()})
    film = film.to(device)
    student.id_encoder = film
    return film


def overlay_canonical_p0_and_empty_head(encoder: nn.Module) -> dict[str, Any]:
    film_path = plan.repo_root() / plan.FILM_STATES_RELATIVE
    head_path = plan.repo_root() / plan.SELECTED_HEAD_RELATIVE
    plan.require(film_path.is_file(), f"missing film_states: {film_path}")
    plan.require(head_path.is_file(), f"missing selected head: {head_path}")
    canonical = torch.load(film_path, map_location="cpu", weights_only=False)
    plan.require("p0" in canonical, "canonical p0 missing")
    encoder.load_state_dict(canonical["p0"], strict=True)
    head_payload = torch.load(head_path, map_location="cpu", weights_only=False)
    head = head_payload["state_dict"]
    head_sha = tensor_state_sha256(head)
    plan.require(
        head_sha == plan.SELECTED_HEAD_STATE_SHA256,
        f"EMPTY head digest drift: {head_sha}",
    )
    full = encoder.state_dict()
    expected_head = {
        "contrast_context.0.weight",
        "contrast_context.0.bias",
        "contrast_film.weight",
        "contrast_film.bias",
    }
    plan.require(set(head) == expected_head, f"EMPTY head key drift: {sorted(head)}")
    for key, value in head.items():
        plan.require(key in full, f"unexpected EMPTY head key {key}")
        full[key] = value
    encoder.load_state_dict(full, strict=True)
    return {
        "head_state_sha256": head_sha,
        "selected_meta": head_payload.get("selection", {}),
        "p0_keys": sorted(canonical["p0"]),
        "empty_means_contrast_input_zero": True,
        "empty_does_not_mean_zero_adapter_weights": True,
    }


def freeze_inherited(student: Any) -> None:
    student.eval()
    if hasattr(student, "freeze_decoder"):
        student.freeze_decoder()
    for parameter in student.parameters():
        parameter.requires_grad = False
    student.decoder.eval()
    student.id_encoder.eval()
    plan.require(bool(getattr(student, "_decoder_frozen", True)), "decoder freeze flag")
    plan.require(
        all(not parameter.requires_grad for parameter in student.parameters()),
        "inherited params not fully frozen",
    )


class FrozenChampion:
    """REF candidate. ``forward_last`` is decoder_raw; Evaluator applies /5."""

    name = "REF"
    training_target_space = plan.TRAINING_TARGET_SPACE

    def __init__(self, student: Any, *, authority: dict[str, Any] | None = None) -> None:
        self.student = student
        self.authority = authority or {}
        freeze_inherited(self.student)

    def forward_last(
        self,
        X: torch.Tensor,
        bank: contracts.SessionBank,
        unit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan.require(X.ndim == 3 and X.shape[1] == plan.WINDOW, "X [B,50,N]")
        identity = bank.E0.to(device=X.device, dtype=X.dtype)
        if identity.ndim == 2:
            identity = identity.unsqueeze(0).expand(X.shape[0], -1, -1)
        mask = unit_mask if unit_mask is not None else bank.unit_mask
        gate = None
        if mask is not None:
            gate = mask.to(device=X.device, dtype=X.dtype).reshape(1, -1, 1)
        # Intentionally no torch.no_grad / inference_mode: decode_with_identity
        # must stay differentiable w.r.t. identity for A later.  Weights are
        # frozen via requires_grad=False and decoder.eval().
        prediction = self.student.decode_with_identity(X, identity, neuron_gate=gate)
        return prediction[:, -1, :]

    def trainable_parameters(self) -> dict[str, torch.nn.Parameter]:
        return {}

    def train(self, mode: bool = True) -> FrozenChampion:
        del mode
        self.student.eval()
        self.student.decoder.eval()
        self.student.id_encoder.eval()
        return self

    def eval(self) -> FrozenChampion:
        return self.train(False)


def load_frozen_champion(*, device: str | torch.device = "cpu") -> FrozenChampion:
    device = torch.device(device)
    model = load_base_lightning_module(device=device)
    student = model.student
    install_empty_film(student, device)
    head_meta = overlay_canonical_p0_and_empty_head(student.id_encoder)
    freeze_inherited(student)
    authority = {
        **cache_key_parts(),
        **head_meta,
        "device": str(device),
        "output_space": OUTPUT_SPACE_CONTRACT,
        "model_setup": "fit_weights_only_no_nwb",
        "data_module_setup_test": False,
        "load_frozen_model_and_data_called": False,
    }
    return FrozenChampion(student, authority=authority)


def empty_contrast_side(move_t4: np.ndarray | torch.Tensor) -> torch.Tensor:
    """``[MOVE-T4, 0]`` per unit. Contrast input is zeros; head weights stay live."""
    t4 = torch.as_tensor(move_t4, dtype=torch.float32)
    plan.require(t4.ndim == 2 and t4.shape[-1] == plan.T4_DIM, "MOVE-T4 [N,4]")
    zeros = torch.zeros(t4.shape[0], CONTRAST_DIM, dtype=torch.float32, device=t4.device)
    return torch.cat([t4, zeros], dim=-1)


def native_e0_and_u(
    encoder: nn.Module,
    trials: torch.Tensor,
    side_t4_plus_contrast: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native chronological FP32 E0 plus per-trial pre_pool ``u``.

    ``trials``: [K, T, N] interpolated 100-bin activity.
    ``side_t4_plus_contrast``: [N, 8] = [MOVE-T4, 0] for EMPTY.
    E0 uses ``reset_stream`` / ``push_trial`` / ``finalize_identity`` only.
    """
    plan.require(trials.ndim == 3, "trials [K,T,N]")
    plan.require(trials.shape[1] == plan.CALIB_TRIAL_LENGTH, "trial length")
    plan.require(trials.shape[2] == plan.CHANNELS, "channels")
    k, _, n_units = trials.shape
    device = trials.device
    dtype = torch.float32
    trials = trials.to(device=device, dtype=dtype)
    side = side_t4_plus_contrast.to(device=device, dtype=dtype)
    plan.require(tuple(side.shape) == (n_units, SIDE_DIM), "side [N,8]")

    encoder.eval()
    u_rows: list[torch.Tensor] = []
    state = encoder.reset_stream(1, n_units, device, dtype)
    state["side_features"] = side.unsqueeze(0)
    with torch.no_grad():
        for trial_idx in range(k):
            trial = trials[trial_idx]
            # pre_pool is Linear(T,64) on [B,N,T]; same operator push_trial uses.
            trial_bnt = trial.unsqueeze(0).permute(0, 2, 1)
            u_rows.append(encoder.pre_pool(trial_bnt).squeeze(0))
            state = encoder.push_trial(state, trial)
        e0 = encoder.finalize_identity(state).squeeze(0)
    u = torch.stack(u_rows, dim=0)
    plan.require(tuple(e0.shape) == (n_units, plan.IDENTITY_DIM), "E0 shape")
    plan.require(tuple(u.shape) == (k, n_units, plan.HIDDEN_DIM), "u shape")
    return e0.detach().cpu().contiguous(), u.detach().cpu().contiguous()


def move_t4_trial_sums(
    calib_neural: np.ndarray,
    calib_trial_change: np.ndarray,
    angles: np.ndarray,
    *,
    session: str,
    start_bin: int = plan.MOVE_T4_START_BIN,
    stop_bin: int = plan.MOVE_T4_STOP_BIN,
    horizon: int = plan.SUPPORT_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uninterpolated mask-filtered bins [5, 30) over the first 33 trials."""
    neural = np.asarray(calib_neural, dtype=np.float32)
    starts = np.flatnonzero(np.asarray(calib_trial_change, dtype=bool))
    ends = np.r_[starts[1:], neural.shape[0]]
    angles = np.asarray(angles, dtype=np.float32)
    plan.require(starts.size >= horizon and angles.size >= horizon, f"{session} lacks M{horizon}")
    sums: list[np.ndarray] = []
    lengths: list[int] = []
    for trial, (start, end) in enumerate(zip(starts[:horizon], ends[:horizon], strict=True)):
        available = int(end - start)
        plan.require(available >= stop_bin, f"{session} trial {trial} has only {available} bins")
        chunk = neural[int(start) + start_bin : int(start) + stop_bin]
        plan.require(chunk.shape == (stop_bin - start_bin, plan.CHANNELS), "MOVE-T4 window shape")
        sums.append(chunk.sum(axis=0, dtype=np.float64))
        lengths.append(stop_bin - start_bin)
    return (
        np.ascontiguousarray(sums, dtype=np.float32),
        np.asarray(lengths, dtype=np.int64),
        np.ascontiguousarray(angles[:horizon], dtype=np.float32),
    )


def fit_move_t4(
    calib_neural: np.ndarray,
    calib_trial_change: np.ndarray,
    angles: np.ndarray,
    *,
    session: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    ensure_streaming_on_path()
    from src.data.falcon_t4_features import t4_from_trial_sums

    sums, lengths, used_angles = move_t4_trial_sums(
        calib_neural, calib_trial_change, angles, session=session
    )
    raw = t4_from_trial_sums(
        sums,
        lengths,
        used_angles,
        source=f"{session}:bins[{plan.MOVE_T4_START_BIN}:{plan.MOVE_T4_STOP_BIN}]",
    )
    values = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    plan.require(values.shape == (plan.CHANNELS, plan.T4_DIM), "normalized MOVE-T4 shape")
    plan.require(bool(np.isfinite(values).all()), "non-finite MOVE-T4")
    return values


def fit_source_move_normalizer(
    per_session: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Source-only MOVE-T4 z-score on the seven held-in sessions.

    Does **not** inherit the old M33 fold normalizer in split_manifest.json.
    """
    ensure_streaming_on_path()
    from src.data.falcon_t4_features import t4_from_trial_sums

    sessions = tuple(plan.HELDIN_SESSIONS)
    plan.require(set(per_session) >= set(sessions), "MOVE-T4 normalizer missing held-in session")
    rows: dict[str, np.ndarray] = {}
    for session in sessions:
        neural, change, angles = per_session[session]
        sums, lengths, used_angles = move_t4_trial_sums(
            neural, change, angles, session=session
        )
        rows[session] = t4_from_trial_sums(
            sums,
            lengths,
            used_angles,
            source=f"{session}:bins[{plan.MOVE_T4_START_BIN}:{plan.MOVE_T4_STOP_BIN}]",
        )
    joined = np.concatenate([rows[session] for session in sessions], axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0).astype(np.float32)
    std[std <= 1.0e-6] = 1.0
    return mean, std, {
        "train_sessions": list(sessions),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "raw_t4_sha256": {session: array_sha256(rows[session]) for session in sessions},
        "inherited_m33_fold_normalizer": False,
        "carrier": "MOVE-T4",
        "bins": [plan.MOVE_T4_START_BIN, plan.MOVE_T4_STOP_BIN],
    }
