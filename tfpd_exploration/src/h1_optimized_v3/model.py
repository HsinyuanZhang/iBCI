"""V3 source-only centered frontend revision; V2 remains immutable."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from tfpd_exploration.src.h1_optimized_v2.model import H1FullWindowControl, H1CurrentQueryDecoder

class _CenteredFrontend:
    """Remove learned zero-input frontend offset before tokenwise affine-free LN."""
    def _z(self,x,bank,dropout_keep=None):
        z=super()._z(x,bank,dropout_keep)
        # Full zero history is explicit: startup therefore maps to exactly zero.
        z0=super()._z(torch.zeros_like(x),bank,dropout_keep)
        return F.layer_norm(z-z0,(z.shape[-1],),weight=None,bias=None,eps=1e-5)
class H1CenteredFull(_CenteredFrontend,H1FullWindowControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.register_buffer('frontend_contract_version',torch.tensor(3,dtype=torch.int64))
class H1CenteredQuery(_CenteredFrontend,H1CurrentQueryDecoder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.register_buffer('frontend_contract_version',torch.tensor(3,dtype=torch.int64))
def make_matched_pair(*,activity_scale=32.):
 a=H1CenteredFull(activity_scale=activity_scale);b=H1CenteredQuery(activity_scale=activity_scale)
 b.frontend.load_state_dict(a.frontend.state_dict());b.final_norm.load_state_dict(a.final_norm.state_dict());b.readout.load_state_dict(a.readout.state_dict());b.temporal.initialize_from_full_window(a.temporal)
 return a,b
