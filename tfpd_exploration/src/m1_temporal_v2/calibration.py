"""Load frozen B3 from S-Fix after Lightning setup. Encoder keys are strict."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import M1_TEMPORAL, REPO_ROOT
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank

from . import bank as v2_bank
from . import plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_experiment_path() -> None:
    experiment = str(REPO_ROOT / "streaming_calibration_exp")
    if experiment not in sys.path:
        sys.path.insert(0, experiment)


def _sfix_state() -> dict[str, Any]:
    if sha256_file(plan.S_FIX_PATH) != plan.S_FIX_SHA256:
        raise RuntimeError("S-Fix bytes drifted")
    payload = torch.load(plan.S_FIX_PATH, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state, dict):
        raise RuntimeError("S-Fix checkpoint has no state_dict")
    return state


def load_frozen_b3_student() -> nn.Module:
    """Create student via setup('fit'), then load id_encoder with strict=True."""
    _ensure_experiment_path()
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import module as fold_module

    state = _sfix_state()
    lit = fold_module.make_module(REPO_ROOT)
    if getattr(lit, "student", None) is not None:
        raise RuntimeError("student existed before setup; refuse implicit init")
    lit.setup("fit")
    student = lit.student
    if student is None:
        raise RuntimeError("S-Fix student missing after setup('fit')")
    encoder_prefix = "student.id_encoder."
    encoder_state = {key[len(encoder_prefix) :]: value for key, value in state.items() if key.startswith(encoder_prefix)}
    if not encoder_state:
        raise RuntimeError("S-Fix checkpoint has no student.id_encoder keys")
    expected = set(student.id_encoder.state_dict())
    incoming = set(encoder_state)
    missing = sorted(expected - incoming)
    unexpected = sorted(incoming - expected)
    if missing or unexpected:
        raise RuntimeError(
            "B3 encoder load is not complete: "
            f"missing={missing[:8]} unexpected={unexpected[:8]} "
            f"n_missing={len(missing)} n_unexpected={len(unexpected)}"
        )
    student.id_encoder.load_state_dict(encoder_state, strict=True)
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student


class FrozenM1Materializer:
    """id_encoder and sealed rSyn3 only. Does not copy S-Fix decoder weights."""

    def __init__(self, student: nn.Module, source_bank: dict[str, Any]) -> None:
        self.student = student
        self.encoder = student.id_encoder
        self.bank = source_bank
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
        carrier = v2_bank.carrier_for_session(self.bank, session)
        t = torch.as_tensor(np.ascontiguousarray(carrier), dtype=torch.float32)
        n_units = int(identity.size(0))
        if tuple(t.shape) != (n_units, 4):
            raise RuntimeError(f"rSyn3 shape {tuple(t.shape)} vs units {n_units}")
        return M1Bank(
            E0=identity.detach().to(dtype=torch.float32).contiguous(),
            T=t,
            unit_mask=torch.ones(n_units, dtype=torch.bool),
        )


def load_frozen_m1_materializer() -> FrozenM1Materializer:
    student = load_frozen_b3_student()
    source_bank = v2_bank.load_source_bank()
    return FrozenM1Materializer(student, source_bank)


def sfix_decoder_keys() -> set[str]:
    state = _sfix_state()
    return {name for name in state if name.startswith("student.decoder.")}
