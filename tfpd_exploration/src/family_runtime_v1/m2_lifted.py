"""Exact M2 constant-slot MHA lift; no approximation or runtime API change."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .m2 import FiveTokenM2Decoder, _FiveTokenExactE
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Error
from .repair import repaired_local_features

@dataclass
class _M2LiftedMHA:
    qwk: torch.Tensor; qbk: torch.Tensor; wv: torch.Tensor; bv: torch.Tensor
    ow: torch.Tensor; ob: torch.Tensor; heads: int; head_dim: int
    @classmethod
    @torch.no_grad()
    def from_mha(cls, mha, slots):
        d,h=mha.embed_dim,mha.num_heads; hd=d//h
        wq,wk,wv=mha.in_proj_weight.chunk(3); bq,bk,bv=mha.in_proj_bias.chunk(3)
        q=F.linear(slots,wq,bq).view(slots.size(0),h,hd).transpose(0,1)
        wk=wk.view(h,hd,d); bk=bk.view(h,hd)
        return cls(torch.matmul(q,wk),torch.einsum('hsd,hd->hs',q,bk),wv.view(h,hd,d).transpose(-1,-2),bv.view(h,1,hd),mha.out_proj.weight,mha.out_proj.bias,h,hd)
    def __call__(self,tokens,keep):
        b,n,d=tokens.shape; s=self.qwk.size(1)
        logits=F.linear(tokens,self.qwk.reshape(self.heads*s,d)).view(b,n,self.heads,s).permute(0,2,3,1)
        logits=(logits+self.qbk[None,:,:,None])*(self.head_dim**-0.5)
        a=torch.softmax(logits.masked_fill(~keep[:,None,None,:],float('-inf')),dim=-1)
        ax=torch.bmm(a.reshape(b,self.heads*s,n),tokens).reshape(b,self.heads,s,d)
        v=torch.matmul(ax,self.wv.unsqueeze(0))+a.sum(-1,keepdim=True)*self.bv.unsqueeze(0)
        return F.linear(v.transpose(1,2).contiguous().reshape(b,s,self.heads*self.head_dim),self.ow,self.ob)

class _LiftedFiveTokenExactE(_FiveTokenExactE):
    @torch.no_grad()
    def __init__(self,model,bank):
        super().__init__(model,bank)
        self._rebuild_lifted()
    @torch.no_grad()
    def _rebuild_lifted(self):
        front=self.model.frontend
        self._lifted=_M2LiftedMHA.from_mha(front.mha,front.slot_norm(front.slots))
        self._lifted_versions=(int(self._lifted.qwk._version),int(self._lifted.qbk._version))
    def _refresh_if_mutated(self):
        # The parent signature audits every model parameter/buffer and active
        # bank tensor.  Re-derive qWk/qbk whenever it observes any mutation:
        # an MHA Q/K, slots, or slot-norm update otherwise leaves stale lift.
        before=self._signature
        super()._refresh_if_mutated()
        if self._signature != before:
            self._rebuild_lifted()
        elif (int(self._lifted.qwk._version),int(self._lifted.qbk._version)) != self._lifted_versions:
            raise RuntimeV3Error("lifted derived qWk/qbk mutated; reset required")
    def static_projection_bytes(self):
        parent=super().static_projection_bytes()
        return parent + self._lifted.qwk.numel()*self._lifted.qwk.element_size() + self._lifted.qbk.numel()*self._lifted.qbk.element_size()
    def _repair_frontend(self,raw):
        front=self.model.frontend; cfg=front.cfg; local=repaired_local_features(front.local_conv,raw)
        b,t,n,_=local.shape; first,activation,second=front.token_mlp
        tokens=front.token_norm(second(activation(F.linear(local,first.weight[:,:16],None)+self.static_token.unsqueeze(1)))).reshape(b*t,n,cfg.set_dim)
        slots=front.slot_norm(front.slots).view(1,1,cfg.slots,cfg.set_dim); q=slots.expand(b,t,-1,-1).reshape(b*t,cfg.slots,cfg.set_dim)
        keep=self.bank.unit_mask
        if keep.ndim==1: keep=keep.unsqueeze(0).expand(b,-1)
        keep=keep.unsqueeze(1).expand(b,t,n).reshape(b*t,n)
        out=q+self._lifted(tokens,keep); out=out+front.slot_ffn(front.slot_ffn_norm(out))
        return front.slot_proj(out.reshape(b,t,cfg.slots*cfg.set_dim))

class LiftedFiveTokenM2Decoder(FiveTokenM2Decoder):
    """Five-token lifecycle with algebraically lifted constant slot attention."""
    def reset(self,dataset_tags):
        super().reset(dataset_tags)
        old=self._engine; new=_LiftedFiveTokenExactE(self.model,old.bank); new.raw=old.raw; new.frontend=old.frontend; self._engine=new
