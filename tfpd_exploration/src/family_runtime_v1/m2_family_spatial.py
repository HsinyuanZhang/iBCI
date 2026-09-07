"""Frozen lifted spatial frontend for actual M2FamilyDecoder explicit attention."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .linear_conv import repaired_local_features_linear
from .grouped_value import grouped_value_projection

class M2FamilyLiftedSpatial:
 @torch.no_grad()
 def __init__(self,model,bank):
  if model.training or next(model.parameters()).device.type!='cpu' or next(model.parameters()).dtype!=torch.float32: raise RuntimeError('CPU float32 eval snapshot required')
  self.model,self.bank,self.front=model,bank,model.frontend; a=self.front.attn; c=self.front.cfg
  if not hasattr(a,'in_proj_weight') or c.conv_kernel!=5: raise TypeError('actual explicit M2 family frontend required')
  self.h,self.d,self.s,self.dim=a.heads,a.head_dim,c.slots,c.set_dim
  w0,w1,w2=self.front.token_mlp[0].weight.split([c.token_in-c.identity_dim-c.t4_dim,c.identity_dim,c.t4_dim],1); self.lw=w0; self.e=F.linear(bank.E0,w1); self.t=F.linear(bank.T,w2); self.b=self.front.token_mlp[0].bias
  qW,kW,vW=a.in_proj_weight.chunk(3); qb,kb,vb=a.in_proj_bias.chunk(3)
  slot=self.front.slot_norm(self.front.slots); q=F.linear(slot,qW,qb).view(self.s,self.h,self.d).transpose(0,1); kw=kW.view(self.h,self.d,self.dim); self.qwk=torch.matmul(q,kw); self.qbk=torch.einsum('hsd,hd->hs',q,kb.view(self.h,self.d)); self.vw=vW.view(self.h,self.d,self.dim).transpose(-1,-2); self.vb=vb.view(self.h,1,self.d); self.slot=slot
  self.route=a.routing(bank.E0,bank.T) if a.routed else None
 def derived_tensors(self):
  return (self.e,self.t,self.slot,self.qwk,self.qbk) + (() if self.route is None else (self.route,))
 def _static(self,x): return x[None,None] if x.ndim==2 else x[:,None]
 @torch.no_grad()
 def encode(self,local):
  B,W,N,_=local.shape; keep=self.bank.unit_mask
  if keep.ndim==1: keep=keep[None].expand(B,-1)
  if keep.shape!=(B,N) or not bool(keep.any(1).all()): raise ValueError('nonempty BxN mask required')
  z=F.linear(local,self.lw)+self._static(self.e)+self._static(self.t)+self.b; tok=self.front.token_norm(self.front.token_mlp[2](F.gelu(z))).reshape(B*W,N,self.dim)
  logits=F.linear(tok,self.qwk.reshape(self.h*self.s,self.dim)).view(B*W,N,self.h,self.s).permute(0,2,3,1); logits=(logits+self.qbk[None,:,:,None])*(self.d**-.5)
  if self.route is not None:
   r=self.route; r=r.expand(B,-1,-1,-1) if r.shape[0]==1 else r
   logits=logits+r[:,None].expand(-1,W,-1,-1,-1).reshape_as(logits)
  pad=(~keep)[:,None].expand(B,W,N).reshape(B*W,N); weight=torch.softmax(logits.masked_fill(pad[:,None,None],float('-inf')),-1)
  attended=torch.bmm(weight.reshape(B*W,self.h*self.s,N),tok).reshape(B*W,self.h,self.s,self.dim); val=grouped_value_projection(attended,self.vw)+weight.sum(-1,keepdim=True)*self.vb[None]
  merged=val.transpose(1,2).reshape(B*W,self.s,self.dim); slots=self.slot[None].expand(B*W,-1,-1)+F.linear(merged,self.front.attn.out_proj_weight,self.front.attn.out_proj_bias); slots=slots+self.front.slot_ffn(self.front.slot_ffn_norm(slots)); return self.front.slot_proj(slots.reshape(B,W,self.s*self.dim))
 @torch.no_grad()
 def full(self,raw): return self.encode(self.front.local_conv(raw))
 @torch.no_grad()
 def repair(self,raw): return self.encode(repaired_local_features_linear(self.front.local_conv,raw))
