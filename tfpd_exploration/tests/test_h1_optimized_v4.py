import torch
import pytest
from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
from tfpd_exploration.src.h1_optimized_v2.model import make_matched_pair as v2pair
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
def bank(n=176): return H1Bank(torch.randn(n,700),torch.randn(n,4),torch.ones(n,dtype=torch.bool))
def test_v4_permutation_mask_and_causality():
 torch.manual_seed(3);m=make_matched_pair()[0].eval();x=torch.randn(1,12,176);b=bank();z=m.encode_frontend(x,b)
 p=torch.randperm(176);bp=H1Bank(b.E0[p],b.T[p],b.unit_mask[p]);assert torch.allclose(z,m.encode_frontend(x[:,:,p],bp),atol=2e-5)
 x2=x.clone();x2[:,8:]+=100;assert torch.allclose(z[:,:8],m.encode_frontend(x2,b)[:,:8],atol=2e-5)
 bad=b.unit_mask.clone();bad[-1]=False;bb=H1Bank(b.E0,b.T,bad);xx=x.clone();xx[:,:,-1]+=999;assert torch.allclose(m.encode_frontend(x,bb),m.encode_frontend(xx,bb),atol=2e-5)
def test_v4_strict_oldstate_rejected_and_optimizer_steps():
 m=make_matched_pair()[0];old=v2pair()[0].state_dict()
 with pytest.raises(RuntimeError):m.load_state_dict(old,strict=True)
 x=torch.randn(1,8,176);y=torch.randn(1,7);o=torch.optim.AdamW(m.parameters(),lr=2e-4);before=m.frontend.phi.weight.detach().clone();o.zero_grad();torch.nn.functional.mse_loss(m.forward_last(x,bank()),y).backward();o.step();assert not torch.equal(before,m.frontend.phi.weight)
@pytest.mark.parametrize('batch_size',[1,4,16])
def test_v4_dropout_effective_mask_is_rowwise_and_nonempty(batch_size):
 torch.manual_seed(31);m=make_matched_pair()[0].eval();x=torch.randn(batch_size,9,176);b=bank();keep=torch.rand(batch_size,176)>.4;keep[:,0]=True
 z=m.encode_frontend(x,b,keep);assert z.shape==(batch_size,9,256);assert torch.isfinite(z).all()
 # Each batched dropout row is exactly its own effective-bank no-dropout run.
 for row in range(batch_size):
  eff=b.unit_mask&keep[row];single=H1Bank(b.E0,b.T,eff)
  assert torch.allclose(z[row:row+1],m.encode_frontend(x[row:row+1],single),atol=2e-5)
 p=torch.randperm(176);bp=H1Bank(b.E0[p],b.T[p],b.unit_mask[p])
 assert torch.allclose(z,m.encode_frontend(x[:,:,p],bp,keep[:,p]),atol=2e-5)
