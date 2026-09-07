"""V4: normalized signed carrier mixing before the matched H1 readers."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank, CausalTransformerStack, _reinit_param
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack

class SignedCarrierFrontend(nn.Module):
    """z=LN(GELU(PW(DW5(sum_i x_i*w_i + pop(x))))), with static normalized w."""
    def __init__(self,cfg=H1_TEMPORAL):
        super().__init__(); self.cfg=cfg; d=cfg.temporal_width
        self.phi=nn.Linear(cfg.e0_dim+cfg.hc_dim,d,bias=False)
        self.pop_projection=nn.Linear(1,d,bias=False)
        self.depthwise=nn.Conv1d(d,d,kernel_size=5,groups=d,bias=False)
        self.pointwise=nn.Linear(d,d,bias=True)
    def weights(self,bank:H1Bank,x,effective_mask=None):
        carrier=torch.cat((bank.E0.to(x),bank.T.to(x)),dim=-1)
        raw=torch.tanh(self.phi(F.layer_norm(carrier,(carrier.shape[-1],),weight=None,bias=None,eps=1e-5)))
        mask=(bank.unit_mask if effective_mask is None else effective_mask).to(x.device,dtype=x.dtype).unsqueeze(-1)
        if mask.ndim==3: raw=raw.unsqueeze(0)
        if effective_mask is None:
            count=mask.sum(0,keepdim=True).clamp_min(1.)
            centered=(raw-(raw*mask).sum(0,keepdim=True)/count)*mask
            return centered/torch.sqrt(centered.square().sum(0,keepdim=True).clamp_min(1e-8))
        count=mask.sum(-2,keepdim=True).clamp_min(1.)
        centered=(raw-(raw*mask).sum(-2,keepdim=True)/count)*mask
        return centered/torch.sqrt(centered.square().sum(-2,keepdim=True).clamp_min(1e-8))
    def forward(self,x,bank,dropout_keep=None):
        bank_mask=bank.unit_mask.to(x.device,dtype=x.dtype)
        if dropout_keep is None:
            # Retain this original no-dropout branch exactly: saved V4 gate
            # checkpoints were trained and evaluated through these operations.
            mask=bank_mask.unsqueeze(0); w=self.weights(bank,x)
            signed=torch.einsum('bwn,nd->bwd',x*mask,w)
            pop=(x*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1.)
        else:
            mask=bank_mask.unsqueeze(0)*dropout_keep.to(x.device,dtype=x.dtype)
            # A dropped unit must be absent from mean-centering and L2 weight
            # normalization, not merely have its neural activity zeroed.
            w=self.weights(bank,x,mask)
            masked_x=x*mask.unsqueeze(1)
            signed=torch.einsum('bwn,bnd->bwd',masked_x,w)
            pop=masked_x.sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).unsqueeze(1).clamp_min(1.)
        mixed=signed+self.pop_projection(pop)
        h=self.depthwise(F.pad(mixed.transpose(1,2),(4,0))).transpose(1,2)
        return F.layer_norm(F.gelu(self.pointwise(h)),(h.shape[-1],),weight=None,bias=None,eps=1e-5)

class _Base(nn.Module):
    prediction_divisor=20.; training_target_space='runtime_scaled_velocity__target_is_20x_native'
    def __init__(self):
        super().__init__();self.cfg=H1_TEMPORAL;self.frontend=SignedCarrierFrontend(self.cfg);self.final_norm=nn.LayerNorm(self.cfg.temporal_width);self.readout=nn.Sequential(nn.Linear(self.cfg.temporal_width,self.cfg.readout_hidden),nn.GELU(),nn.Linear(self.cfg.readout_hidden,self.cfg.out_dim));self.register_buffer('frontend_contract_version',torch.tensor(4,dtype=torch.int64))
    def encode_frontend(self,x,bank,dropout_keep=None): return self.frontend(x,bank,dropout_keep)
    def forward_last(self,x,bank,dropout_keep=None): return self.readout(self.final_norm(self.forward_hidden_from_frontend(self.encode_frontend(x,bank,dropout_keep))))
class H1SignedFull(_Base):
    def __init__(self):
        super().__init__();self.temporal=CausalTransformerStack(self.cfg)
    def forward_hidden_from_frontend(self,z): return self.temporal(z)[:,-1]
class H1SignedQuery(_Base):
    def __init__(self):
        super().__init__();self.temporal=QueryTemporalStack(width=self.cfg.temporal_width,heads=self.cfg.heads,layers=self.cfg.layers,ffn=self.cfg.ffn,window=self.cfg.window,age_buckets=16,seed=42)
        for b in self.temporal.blocks:b.age_bias.data.zero_()
    def forward_hidden_from_frontend(self,z): return self.temporal(z,use_checkpoint=False).squeeze(1)
def make_matched_pair(*,activity_scale=1.):
    if activity_scale!=1.:raise ValueError('V4 raw-count frontend contract fixes activity_scale=1')
    a,b=H1SignedFull(),H1SignedQuery(); g=torch.Generator(device='cpu').manual_seed(42+4_000_003)
    for n,p in sorted(a.named_parameters()):_reinit_param(n,p,g)
    b.frontend.load_state_dict(a.frontend.state_dict());b.final_norm.load_state_dict(a.final_norm.state_dict());b.readout.load_state_dict(a.readout.state_dict());b.temporal.initialize_from_full_window(a.temporal)
    return a,b
