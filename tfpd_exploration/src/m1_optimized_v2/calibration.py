"""Strict B3 identity loader for the new M1 adapter.

Only `student.id_encoder` is consumed.  Decoder keys are audited but never
copied into this decoder, which keeps B3 identity provenance explicit.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import M1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank
from . import bank, plan

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()
def _streaming_path():
    p=str(plan.REPO_ROOT / "streaming_calibration_exp")
    if p not in sys.path: sys.path.insert(0,p)
def load_frozen_b3() -> nn.Module:
    """Setup Lightning first, then strict-load exactly all B3 encoder keys."""
    _streaming_path()
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import module
    if _sha256(plan.S_FIX_PATH) != plan.S_FIX_SHA256: raise RuntimeError("frozen B3 checkpoint checksum drift")
    state=torch.load(plan.S_FIX_PATH,map_location="cpu",weights_only=False)
    state=state["state_dict"] if isinstance(state,dict) and "state_dict" in state else state
    lit=module.make_module(plan.REPO_ROOT)
    if getattr(lit,"student",None) is not None: raise RuntimeError("B3 student unexpectedly initialized before setup")
    lit.setup("fit")
    student=lit.student
    if student is None: raise RuntimeError("B3 student missing after setup('fit')")
    prefix="student.id_encoder."; incoming={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    expected=set(student.id_encoder.state_dict()); missing=sorted(expected-set(incoming)); unexpected=sorted(set(incoming)-expected)
    if not incoming or missing or unexpected: raise RuntimeError(f"strict B3 id_encoder keys failed: missing={missing[:5]} unexpected={unexpected[:5]}")
    student.id_encoder.load_state_dict(incoming,strict=True)
    # Assert the checkpoint had decoder keys, but deliberately do not load or retain them.
    if not any(k.startswith("student.decoder.") for k in state): raise RuntimeError("B3 checkpoint lacks decoder provenance keys")
    student.eval()
    for p in student.parameters(): p.requires_grad_(False)
    return student

class FrozenMaterializer:
    def __init__(self, student: nn.Module, source_bank: dict[str,Any]): self.student,self.source_bank=student,source_bank
    @torch.no_grad()
    def materialize(self, calib_trials: torch.Tensor, session: str) -> M1Bank:
        identity=self.student.compute_identity(calib_trials,side_features=None)
        if identity.dim()==3: identity=identity[0]
        if identity.shape != (plan.N_UNITS,M1_TEMPORAL.e0_dim): raise RuntimeError(f"B3 identity shape drift: {tuple(identity.shape)}")
        carrier=torch.as_tensor(np.ascontiguousarray(self.source_bank["normalized"][session]),dtype=torch.float32)
        if carrier.shape != (plan.N_UNITS,4): raise RuntimeError(f"carrier shape drift: {tuple(carrier.shape)}")
        return M1Bank(E0=identity.float().contiguous(),T=carrier,unit_mask=torch.ones(plan.N_UNITS,dtype=torch.bool))
def load_materializer() -> FrozenMaterializer: return FrozenMaterializer(load_frozen_b3(),bank.load())
