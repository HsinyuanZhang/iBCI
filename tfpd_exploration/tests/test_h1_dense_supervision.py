from __future__ import annotations
import numpy as np,torch,pytest
from tfpd_exploration.src.h1_family_v1 import dense_supervision as d

class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.final_norm=torch.nn.Identity();self.readout=torch.nn.Linear(7,7,bias=False)
  with torch.no_grad():self.readout.weight.copy_(torch.eye(7))
 def encode_frontend(self,x,bank,dropout_keep=None):return x[:,:,:7]
 def temporal(self,z):return torch.cumsum(z,1) # causal
 def forward_last(self,x,bank,dropout_keep=None):return self.readout(self.final_norm(self.temporal(self.encode_frontend(x,bank,dropout_keep))[:,-1]))

def test_last_injectable_position_exactly_matches_forward_last_and_is_causal():
 m=Tiny();x=torch.randn(2,700,176); got=d.assert_last_parity(m,x,None)
 for pos in d.POSITIONS:
  changed=x.clone();changed[:,pos+1:]+=100
 assert torch.equal(d.forward_positions(m,x,None,positions=(pos,)),d.forward_positions(m,changed,None,positions=(pos,)))
 target=torch.randn(2,1,7);dense=d.weighted_loss(d.forward_positions(m,x,None,positions=(699,)),target,torch.ones(2,1,dtype=torch.bool),positions=(699,)); direct=torch.nn.functional.mse_loss(m.forward_last(x,None),target[:,0]);torch.testing.assert_close(dense,direct,atol=1e-5,rtol=1e-6)
 assert got.shape==(2,7)
def test_targets_valid_mask_and_weighted_loss_endpoint_requirement():
 velocity=np.arange(1000*7,dtype=np.float32).reshape(1000,7);mask=np.ones(1000,bool);mask[466]=False
 target,valid=d.dense_targets({'velocity':velocity,'eval_mask':mask},np.array([0],np.int64))
 assert valid.tolist()==[[True,False,True,True]] and torch.equal(target[0,0],torch.from_numpy(velocity[349]*20))
 loss=d.weighted_loss(torch.zeros_like(target),target,valid);assert torch.isfinite(loss)
 with pytest.raises(RuntimeError,match='endpoint'):d.dense_targets({'velocity':velocity,'eval_mask':np.zeros(1000,bool)},np.array([0],np.int64))
def test_route_like_gate_receives_gradient():
 m=Tiny();m.readout.weight.requires_grad_(True);x=torch.randn(2,700,176);p=d.forward_positions(m,x,None);t=torch.zeros_like(p);loss=d.weighted_loss(p,t,torch.ones(2,4,dtype=torch.bool));loss.backward();assert m.readout.weight.grad.abs().sum()>0

def test_real_tiny_h1_pair_dense_causality_parity_state_and_route_gate_gradient():
 from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1TemporalConfig
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
 cfg=H1TemporalConfig(conv_channels=4,conv_kernel=3,e0_dim=8,hc_dim=4,set_dim=8,slots=2,heads=2,layers=1,temporal_width=8,ffn=16,readout_hidden=8,n_units=4,route_key_dim=2)
 _flat,route=make_v2_unscaled_dot_localbalanced_pair(seed=42,cfg=cfg);route.eval();x=torch.randn(1,700,4);bank=H1Bank(torch.randn(4,8),torch.randn(4,4),torch.ones(4,dtype=torch.bool));before={k:v.detach().clone() for k,v in route.state_dict().items()}
 torch.testing.assert_close(d.assert_last_parity(route,x,bank),route.forward_last(x,bank),atol=0,rtol=0)
 for pos in d.POSITIONS:
  altered=x.clone();altered[:,pos+1:]+=1
  torch.testing.assert_close(d.forward_positions(route,x,bank,positions=(pos,)),d.forward_positions(route,altered,bank,positions=(pos,)),atol=1e-6,rtol=1e-6)
 assert all(torch.equal(value,route.state_dict()[key]) for key,value in before.items())
 route.train();prediction=d.forward_positions(route,x,bank);loss=d.weighted_loss(prediction,torch.zeros_like(prediction),torch.tensor([[True,False,True,True]]));loss.backward()
 assert route.frontend.attn.g.grad is not None and route.frontend.attn.g.grad.abs().sum()>0
