"""Frozen M1 B3 identity + rSyn3 M10. Does not train S-Fix decoder weights."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import sys

from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import (
    M1_TEMPORAL,
    REPO_ROOT,
    S_FIX_PATH,
    S_FIX_SHA256,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenM1Materializer:
    """id_encoder and rSyn3 only. New decoder must not copy S-Fix decoder weights."""

    def __init__(self, student: nn.Module, bank: dict[str, Any]) -> None:
        self.student = student
        self.encoder = student.id_encoder
        self.bank = bank
        self.e0_is_fused_identity = True
        self.student.eval()
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def materialize_bank(self, calib_trials: torch.Tensor, session: str) -> M1Bank:
        identity = self.student.compute_identity(calib_trials, side_features=None)
        if identity.dim() == 3:
            identity = identity[0]
        if tuple(identity.shape[-1:]) != (M1_TEMPORAL.e0_dim,):
            raise RuntimeError(f"E0 dim {tuple(identity.shape)} != [N,{M1_TEMPORAL.e0_dim}]")
        from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import carrier_bank

        carrier = carrier_bank.carrier_for_arm(self.bank, session, "S-Fix")
        t = torch.as_tensor(np.ascontiguousarray(carrier), dtype=torch.float32)
        n_units = int(identity.size(0))
        if tuple(t.shape) != (n_units, 4):
            raise RuntimeError(f"rSyn3 shape {tuple(t.shape)} vs units {n_units}")
        return M1Bank(
            E0=identity.detach().to(dtype=torch.float32).contiguous(),
            T=t,
            unit_mask=torch.ones(n_units, dtype=torch.bool),
        )


def _ensure_experiment_path() -> None:
    experiment = str(REPO_ROOT / "streaming_calibration_exp")
    if experiment not in sys.path:
        sys.path.insert(0, experiment)


def _load_sfix_student() -> nn.Module:
    _ensure_experiment_path()
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import module as fold_module

    if sha256_file(S_FIX_PATH) != S_FIX_SHA256:
        raise RuntimeError("S-Fix bytes drifted")
    payload = torch.load(S_FIX_PATH, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    lit = fold_module.make_module(REPO_ROOT)
    student = lit.student
    if student is None:
        raise RuntimeError("S-Fix student missing")
    missing, unexpected = student.load_state_dict(
        {k[len("student.") :]: v for k, v in state.items() if k.startswith("student.")},
        strict=False,
    )
    del missing, unexpected
    return student


def load_frozen_m1_materializer() -> FrozenM1Materializer:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import carrier_bank

    student = _load_sfix_student()
    bank = carrier_bank.build_fold0_carrier_bank(REPO_ROOT)
    return FrozenM1Materializer(student, bank)


def sfix_decoder_keys() -> set[str]:
    payload = torch.load(S_FIX_PATH, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    return {name for name in state if name.startswith("student.decoder.")}
