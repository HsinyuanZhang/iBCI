"""The only three APFC trainable gate families."""
from __future__ import annotations
import hashlib, math
from typing import Any
import torch
from torch import nn
from . import plan

class GateError(RuntimeError): pass
def _need(ok,msg):
    if not ok: raise GateError(msg)

def dct4_basis() -> torch.Tensor:
    """First four orthonormal DCT-II vectors, constructed in float64."""
    t=torch.arange(plan.WINDOW_BINS,dtype=torch.float64).unsqueeze(1)
    k=torch.arange(4,dtype=torch.float64).unsqueeze(0)
    values=torch.cos(math.pi/plan.WINDOW_BINS*(t+0.5)*k)*math.sqrt(2.0/plan.WINDOW_BINS)
    values[:,0]=1.0/math.sqrt(plan.WINDOW_BINS)
    _need(torch.allclose(values.T@values,torch.eye(4,dtype=torch.float64),atol=1e-12,rtol=0), 'APFC DCT orthogonality drift')
    return values

def dct_sha256() -> str:
    return hashlib.sha256(dct4_basis().contiguous().numpy().tobytes()).hexdigest()

class CapacityGate(nn.Module):
    def __init__(self, arm: str, *, source_fit_mean: float=0., source_fit_std: float=1.) -> None:
        super().__init__(); _need(arm in plan.ARMS,'APFC arm unknown'); self.arm=arm
        if arm=='A-S1': self.params=nn.Parameter(torch.zeros(1,dtype=torch.float32))
        elif arm=='A-TB4':
            self.params=nn.Parameter(torch.zeros(4,dtype=torch.float32)); self.register_buffer('basis',dct4_basis().to(torch.float32),persistent=True)
        else: self.params=nn.Parameter(torch.zeros(2,dtype=torch.float32))
        self.source_fit_mean=float(source_fit_mean); self.source_fit_std=float(source_fit_std)
        _need(self.source_fit_std>0 and math.isfinite(self.source_fit_std),'APFC DC2 standardizer invalid')
    def zero_exact(self)->bool:
        return bool(torch.all(self.params.detach()==0).item() and not bool(torch.signbit(self.params.detach()).any().item()))
    def gate(self, *, z: torch.Tensor|None, native: torch.Tensor, post: torch.Tensor, training: bool) -> torch.Tensor:
        _need(native.shape==post.shape and native.ndim==3 and native.shape[-1]==plan.WINDOW_BINS,'APFC branch geometry drift')
        if not training and self.zero_exact(): return native
        if self.arm=='A-S1': q=self.params[0].view(1,1,1)
        elif self.arm=='A-TB4': q=(self.basis@self.params).view(1,1,plan.WINDOW_BINS)
        else:
            _need(z is not None and z.numel()==1 and torch.isfinite(z).all(),'APFC DC2 state statistic missing')
            q=(self.params[0]+self.params[1]*z).view(1,1,1)
        return native+torch.tanh(q)*(post-native)
    def evidence(self)->dict[str,object]:
        return {'arm':self.arm,'parameter_count':int(self.params.numel()),'parameters_shape':list(self.params.shape),
                'zero_exact_positive':self.zero_exact(),'dct_sha256':dct_sha256() if self.arm=='A-TB4' else None,
                'source_fit_mean':self.source_fit_mean if self.arm=='A-DC2' else None,
                'source_fit_std':self.source_fit_std if self.arm=='A-DC2' else None}

def disagreement_statistic(values: torch.Tensor)->torch.Tensor:
    """Label-free s on `[M,N,50]` post-fusion per-trial identities."""
    _need(values.ndim==3 and values.shape[0]>=1 and values.shape[-1]==plan.WINDOW_BINS,'APFC DC2 value geometry drift')
    mean=values.mean(dim=0,keepdim=True); d=torch.sqrt(torch.mean((values-mean)**2)); r=torch.sqrt(torch.mean(values**2))
    out=torch.log((d+1e-8)/(r+1e-8)); _need(torch.isfinite(out).all(),'APFC DC2 nonfinite statistic'); return out

__all__=('CapacityGate','GateError','dct4_basis','dct_sha256','disagreement_statistic')
