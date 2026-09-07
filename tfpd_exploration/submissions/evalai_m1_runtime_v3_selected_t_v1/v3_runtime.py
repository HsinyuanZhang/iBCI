"""Exact finite-W current-query inference with explicit immutable lifecycle.

This is deliberately separate from the reference stream.  It supports one
calibration bank per batch row, while preserving the reference's W=100/k=5
left-boundary repair law.  Cached state is valid only between explicit
``refresh_state`` calls; detected mutation fails closed rather than reusing
stale frontend/KV values.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Sequence
import numpy as np
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class BankBatch:
    """Batch-indexed [B,N,*] bank whose rows match neural input rows exactly."""
    E0: torch.Tensor
    T: torch.Tensor
    unit_mask: torch.Tensor
    session_ids: tuple[Hashable, ...]
    unit_ids: tuple[tuple[Hashable, ...], ...]

    def validate(self, batch: int, units: int) -> None:
        if self.E0.ndim != 3 or self.T.ndim != 3 or self.unit_mask.ndim != 2:
            raise ValueError("heterogeneous bank tensors must be [B,N,*], [B,N,*], [B,N]")
        if self.E0.shape[:2] != (batch, units) or self.T.shape[:2] != (batch, units) or self.unit_mask.shape != (batch, units):
            raise ValueError("bank batch/unit dimensions must exactly match stream rows")
        if len(self.session_ids) != batch or len(self.unit_ids) != batch:
            raise ValueError("one explicit session/unit roster is required per batch row")
        if any(len(r) != units or len(set(r)) != units for r in self.unit_ids):
            raise ValueError("each row must carry a unique, full unit roster")
        if self.unit_mask.dtype != torch.bool or not bool(self.unit_mask.any(dim=1).all()):
            raise ValueError("every batch row must have a nonempty bool unit mask")

def _token(t: torch.Tensor) -> tuple:
    try: version=t._version
    except RuntimeError: version=None
    return (t.data_ptr(), version, tuple(t.shape), str(t.dtype), str(t.device))

def _unique_storage_bytes(tensors, *, excluded=()) -> int:
    """Count each persistent storage once, never charging shared model storage."""
    seen=set()
    for value in excluded:
        if isinstance(value,torch.Tensor):
            seen.add((value.device, value.untyped_storage().data_ptr()))
    total=0
    for value in tensors:
        if not isinstance(value,torch.Tensor): continue
        storage=value.untyped_storage(); key=(value.device,storage.data_ptr())
        if key in seen: continue
        seen.add(key); total+=storage.nbytes()
    return total

class HeterogeneousCurrentQueryStream:
    """B independent rows, static hetero banks, and no hidden mutation reuse."""
    def __init__(self, model, bank: BankBatch, *, window: int | None=None, kernel: int=5):
        if model.training: raise ValueError("model must be eval before stream construction")
        if getattr(model, "routed", False): raise ValueError("v3 static frontend is authorized only for current_query routed=False")
        self.model=model; self.window=int(model.cfg.window); self.kernel=int(kernel)
        if (window is not None and int(window) != self.window) or self.window != 100 or self.kernel != 5 or self.kernel != int(model.cfg.conv_kernel): raise ValueError("fixed M1 W=100/k=5 contract drift")
        self.bank=bank; self.batch=bank.E0.shape[0]; self.units=bank.E0.shape[1]; bank.validate(self.batch,self.units)
        self.raw=self.z=None; self.memories=[]; self._static={}; self._lifecycle=None
        self.refresh_state(bank=bank)

    def _lifecycle_token(self):
        return (id(self.bank), tuple(self.bank.session_ids), tuple(self.bank.unit_ids), tuple((n,_token(p)) for n,p in self.model.named_parameters()), tuple((n,_token(b)) for n,b in self.model.named_buffers()), _token(self.bank.E0),_token(self.bank.T),_token(self.bank.unit_mask))
    def _check_immutable(self):
        if self.model.training: raise RuntimeError("model entered training mode; call refresh_state after restoring eval")
        if self._lifecycle != self._lifecycle_token(): raise RuntimeError("model/bank/dtype changed: fail closed; call refresh_state explicitly")
    def _ref(self): return next(self.model.parameters())

    @torch.no_grad()
    def refresh_state(self, *, bank: BankBatch | None=None, history: torch.Tensor | None=None) -> None:
        """The only authorized invalidation/rebuild point after bank/model changes."""
        if bank is not None: self.bank=bank
        ref=self._ref(); self.batch=self.bank.E0.shape[0]; self.units=self.bank.E0.shape[1]; self.bank.validate(self.batch,self.units)
        if self.model.training: raise ValueError("refresh requires eval model")
        if self.bank.E0.device != ref.device or self.bank.T.device != ref.device or self.bank.unit_mask.device != ref.device: raise ValueError("bank and model must already share one device")
        if self.bank.E0.dtype != ref.dtype or self.bank.T.dtype != ref.dtype: raise ValueError("bank/model dtype mismatch")
        checked = (*self.model.parameters(), *self.model.buffers(), self.bank.E0, self.bank.T, self.bank.unit_mask)
        if any(_token(value)[1] is None for value in checked):
            raise ValueError("versionless inference tensors are unsupported; construct outside inference_mode")
        raw=torch.zeros(self.batch,self.window,self.units,device=ref.device,dtype=ref.dtype)
        self.observation_counts=torch.zeros(self.batch,device=ref.device,dtype=torch.long)
        if history is not None:
            h=torch.as_tensor(history,device=ref.device,dtype=ref.dtype)
            if h.ndim!=3 or h.shape[0]!=self.batch or h.shape[2]!=self.units or not bool(torch.isfinite(h).all()): raise ValueError("history must be finite [B,T,N]")
            n=min(self.window,h.shape[1])
            if n:
                raw[:,-n:]=h[:,-n:]
                self.observation_counts.fill_(n)
        self.raw=raw; self._build_static(); self.z=self._frontend(raw); self.memories=self._project(self.z); self._lifecycle=self._lifecycle_token()

    @torch.no_grad()
    def _build_static(self):
        """Cache only input-independent affine/slot/age work under lifecycle token."""
        f=self.model.frontend; m=f.token_mlp; c=f.cfg; wloc,we0,whc=m.fc1.weight.split([c.conv_channels,c.e0_dim,c.hc_dim],dim=1)
        affine=F.linear(self.bank.E0,we0,None)+F.linear(self.bank.T,whc,None)+m.fc1.bias
        slots=f.slot_norm(f.slots).view(1,c.slots,c.set_dim)
        a=f.attn; q=a.q_proj(slots).view(1,c.slots,a.n_heads,a.head_dim).transpose(1,2)
        ages=torch.arange(self.window-1,-1,-1,device=affine.device); buckets=torch.div(ages*16,self.window,rounding_mode='floor').clamp(max=15)
        biases=[block.age_bias[:,buckets].to(dtype=affine.dtype).unsqueeze(0).unsqueeze(2) for block in self.model.temporal.blocks]
        self._static={'wloc':wloc,'affine':affine,'slots':slots,'q':q,'age_biases':biases}

    @torch.no_grad()
    def _frontend(self,x, *, rows=None):
        f=self.model.frontend; local=f.local_conv(x); b,t,n,_=local.shape; a=f.attn
        affine=self._static['affine'] if rows is None else self._static['affine'][rows]
        keep=self.bank.unit_mask if rows is None else self.bank.unit_mask[rows]
        hidden=F.linear(local,self._static['wloc'],None)+affine.unsqueeze(1)
        tokens=f.token_norm(f.token_mlp.fc2(F.gelu(hidden)))
        query=self._static['slots'].unsqueeze(1).expand(b,t,-1,-1).reshape(b*t,f.cfg.slots,f.cfg.set_dim)
        # q is constant and exactly equals q_proj(slot_norm(slots)); only expand is dynamic.
        q=self._static['q'].unsqueeze(1).expand(b,t,-1,-1,-1).reshape(b*t,a.n_heads,f.cfg.slots,a.head_dim)
        key=a.k_proj(tokens).view(b*t,n,a.n_heads,a.head_dim).transpose(1,2); value=a.v_proj(tokens).view(b*t,n,a.n_heads,a.head_dim).transpose(1,2)
        logits=torch.matmul(q,key.transpose(-2,-1))*(a.head_dim**-0.5)
        pad=(~keep).unsqueeze(1).expand(-1,t,-1).reshape(b*t,n)
        weights=torch.nan_to_num(torch.softmax(logits.masked_fill(pad[:,None,None,:],float('-inf')),dim=-1),nan=0.0)
        attended=a.out_proj(torch.matmul(weights,value).transpose(1,2).contiguous().view(b*t,f.cfg.slots,f.cfg.set_dim))
        out=query+attended; out=out+f.slot_ffn(f.slot_ffn_norm(out))
        return f.slot_proj(out.reshape(b,t,f.cfg.slots*f.cfg.set_dim))

    @torch.no_grad()
    def _project(self,z): return [block.project_memory(z) for block in self.model.temporal.blocks]
    @torch.no_grad()
    def _advance_memory(self,changed, active):
        updates=self._project(self.z[:active,changed]); nxt=[]
        for (k,v),(uk,uv) in zip(self.memories,updates,strict=True):
            nk,nv=k.clone(),v.clone(); nk[:active,:,:-1]=k[:active,:,1:]; nv[:active,:,:-1]=v[:active,:,1:]; nk[:active,:,changed]=uk; nv[:active,:,changed]=uv; nxt.append((nk,nv))
        self.memories=nxt

    @torch.no_grad()
    def observe(self, observations, *, active: int | None=None) -> None:
        self._check_immutable(); x=torch.as_tensor(observations,device=self.raw.device,dtype=self.raw.dtype)
        active=self.batch if active is None else int(active)
        if not 0<active<=self.batch or x.ndim!=2 or x.shape!=(active,self.units) or not bool(torch.isfinite(x).all()): raise ValueError("observations must be finite [active,N], 1<=active<=B")
        # A partial evaluator batch is an observation gap, not a padded zero or
        # repeated bin: every inactive lane's complete causal state is frozen.
        raw=self.raw.clone(); raw[:active,:-1]=self.raw[:active,1:]; raw[:active,-1]=x
        left=self._frontend(raw[:active,:self.kernel-1], rows=slice(0,active)); right=self._frontend(raw[:active,-self.kernel:], rows=slice(0,active))[:,-1:]
        z=self.z.clone(); z[:active,:self.kernel-1]=left; z[:active,self.kernel-1:-1]=self.z[:active,self.kernel:]; z[:active,-1:]=right
        self.raw,self.z=raw,z; self._advance_memory(tuple(range(self.kernel-1))+(self.window-1,),active); self.observation_counts[:active]+=1

    @torch.no_grad()
    def current_prediction(self):
        self._check_immutable(); q=self.z[:,-1:]
        for block,memory,bias in zip(self.model.temporal.blocks,self.memories,self._static['age_biases'],strict=True):
            k,v=memory; h=block.heads; d=block.head_dim; qq=block.q_proj(block.query_norm(q)).reshape(self.batch,1,h,d).transpose(1,2)
            ctx=F.scaled_dot_product_attention(qq,k,v,attn_mask=bias,dropout_p=0.0,is_causal=False).transpose(1,2).reshape(self.batch,1,block.width)
            q=q+block.out_proj(ctx); q=q+block.ffn(block.ffn_norm(q))
        return self.model.readout(self.model.final_norm(q))[:,0]
    @torch.no_grad()
    def predict_tensor(self, observations, *, active: int | None=None):
        """Internal tensor-only inference used for parity and profiling internals."""
        self.observe(observations,active=active); out=self.current_prediction(); return out if active is None else out[:int(active)]
    @torch.no_grad()
    def predict(self, observations, *, active: int | None=None):
        """Public API: direct C-contiguous float32 host input and copied NumPy output."""
        value=np.asarray(observations)
        active=value.shape[0] if active is None and value.ndim else (self.batch if active is None else int(active))
        if value.dtype != np.float32 or value.ndim != 2 or value.shape != (active,self.units) or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError("public observations must be finite C-contiguous float32 [active,N]")
        output=self.predict_tensor(torch.from_numpy(value),active=active).detach().cpu().numpy().astype(np.float32,copy=True)
        if output.shape != (active,self.model.cfg.out_dim) or not np.isfinite(output).all(): raise RuntimeError("non-finite or malformed public prediction")
        return output
    def on_done(self, dones=None): return None
    @property
    def state_bytes(self):
        tensors=[self.raw,self.z,self.observation_counts,*[x for kv in self.memories for x in kv]]
        return _unique_storage_bytes(tensors)+self.static_cache_bytes
    @property
    def rolling_state_bytes(self):
        tensors=[self.raw,self.z,self.observation_counts,*[x for kv in self.memories for x in kv]]
        return _unique_storage_bytes(tensors)

    @property
    def static_cache_bytes(self):
        """Owned immutable-bank derivatives, excluding views into shared model weights."""
        static=(self._static.get('affine'),self._static.get('slots'),self._static.get('q'),*self._static.get('age_biases', []))
        model_storage=(*self.model.parameters(),*self.model.buffers())
        return _unique_storage_bytes(static,excluded=model_storage)
