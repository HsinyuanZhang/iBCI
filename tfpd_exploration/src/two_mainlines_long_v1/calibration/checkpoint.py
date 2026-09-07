"""Real disk save/load: AdamW, sampler cursor, CPU/CUDA RNG, normalizer arrays."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import constants as C
from .factory import FreshPArm, named_arm_parameters
from . import optimizer as optmod


class CheckpointError(RuntimeError):
    """Fail closed for new-stage resume."""


def _rng_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state()
    return payload


def _restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(payload["torch"])
    np.random.set_state(payload["numpy"])
    random.setstate(payload["python"])
    if "cuda" in payload:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise CheckpointError("checkpoint has CUDA RNG but restore device is not CUDA")
        torch.cuda.set_rng_state(payload["cuda"])


def save_arm(arm: FreshPArm, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opt_state = arm.optimizer.state_dict()
    if not opt_state.get("state"):
        raise CheckpointError("AdamW state is empty; take at least one real step before save")
    frozen = arm.frozen
    payload = {
        "schema": "two_mainlines_p_new_stage_v1",
        "arm": arm.name,
        "model": copy.deepcopy(arm.student.state_dict()),
        "basis": copy.deepcopy(arm.basis.state_dict()),
        "optimizer": copy.deepcopy(opt_state),
        "rng": _rng_payload(arm.device),
        "sampler": {
            "seed": C.FORMAL_SEED if arm.name != "profile" else C.PROFILE_SEED,
            "index": int(arm.sampler_index),
            "epoch": int(arm.sampler_epoch),
            "global_step": int(arm.global_step),
            "inherits_smoke_rng": False,
        },
        "normalizer": {
            "mu0": np.ascontiguousarray(frozen.mu0),
            "sigma0": np.ascontiguousarray(frozen.sigma0),
            "d0": np.ascontiguousarray(frozen.d0),
            "scale": np.ascontiguousarray(frozen.scale),
            "receipt": frozen.receipt(),
        },
        "train_mode": optmod.record_train_mode(arm.student, arm.basis, arm.optimizer),
    }
    torch.save(payload, path)
    return path


def load_into_arm(arm: FreshPArm, path: Path) -> FreshPArm:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "basis", "optimizer", "rng", "sampler", "normalizer"}
    missing = required - set(payload)
    if missing:
        raise CheckpointError(f"new-stage fields missing: {sorted(missing)}")
    opt_state = payload["optimizer"]
    if not opt_state.get("state"):
        raise CheckpointError("restored AdamW state is empty")
    sampler = payload["sampler"]
    if int(sampler.get("index", 0)) == 0 and int(sampler.get("global_step", 0)) == 0:
        raise CheckpointError("sampler cursor is still the dummy zero index")
    arrays = payload["normalizer"]
    for key in ("mu0", "sigma0", "d0", "scale"):
        if key not in arrays:
            raise CheckpointError(f"normalizer array {key} missing")
    if not np.array_equal(np.asarray(arrays["mu0"]), arm.frozen.mu0):
        raise CheckpointError("restored mu0 drifted from frozen normalizer")
    if not np.array_equal(np.asarray(arrays["sigma0"]), arm.frozen.sigma0):
        raise CheckpointError("restored sigma0 drifted from frozen normalizer")
    arm.student.load_state_dict(payload["model"])
    arm.basis.load_state_dict(payload["basis"])
    named = named_arm_parameters(arm.student, arm.basis, train_basis=arm.train_basis)
    arm.optimizer = optmod.build_adamw(named, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    arm.optimizer.load_state_dict(opt_state)
    arm.sampler_index = int(sampler["index"])
    arm.sampler_epoch = int(sampler["epoch"])
    arm.global_step = int(sampler["global_step"])
    _restore_rng(payload["rng"], arm.device)
    arm.enter_train()
    return arm


def adamw_nonempty(optimizer: torch.optim.AdamW) -> bool:
    return bool(optimizer.state_dict().get("state"))
