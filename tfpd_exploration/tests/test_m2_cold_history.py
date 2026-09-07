import torch,pytest
from tfpd_exploration.src.m2_family_v1.cold_history import apply_prefix_dropout
def test_shape_alias_and_determinism_global_rng():
 x=torch.randn(8,50,96);before=x.clone();torch.manual_seed(9);a=torch.rand(3);y,l=apply_prefix_dropout(x,seed=1,epoch=2,batch_id=3);b=torch.rand(3);torch.manual_seed(9);assert torch.equal(a,torch.rand(3)) and torch.equal(b,torch.rand(3));assert torch.equal(x,before) and y.data_ptr()!=x.data_ptr();y2,l2=apply_prefix_dropout(x,seed=1,epoch=2,batch_id=3);assert torch.equal(y,y2) and torch.equal(l,l2)
 for i,L in enumerate(l.tolist()):assert torch.equal(y[i,50-L:],x[i,50-L:]) and torch.equal(y[i,:50-L],torch.zeros_like(y[i,:50-L]))
 _,other=apply_prefix_dropout(x,seed=1,epoch=2,batch_id=4);assert not torch.equal(l,other)
def test_control_and_invalid():
 x=torch.randn(2,50,96);y,l=apply_prefix_dropout(x,seed=0,epoch=0,batch_id=0,probability=0.);assert torch.equal(x,y) and y.data_ptr()!=x.data_ptr() and torch.equal(l,torch.tensor([50,50]))
 for bad in (torch.randn(2,49,96),torch.randn(2,50,96,dtype=torch.float64)):
  with pytest.raises(ValueError):apply_prefix_dropout(bad,seed=0,epoch=0,batch_id=0)
 with pytest.raises(ValueError):apply_prefix_dropout(x,seed=0,epoch=0,batch_id=0,probability=.2)
 for p in (0.,.5):
  with pytest.raises(ValueError):apply_prefix_dropout(x,seed=-1,epoch=0,batch_id=0,probability=p)
 with pytest.raises(ValueError):apply_prefix_dropout(torch.empty(0,50,96),seed=0,epoch=0,batch_id=0)
