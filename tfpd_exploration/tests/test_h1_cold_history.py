import pytest, torch
from tfpd_exploration.src.h1_family_v1.cold_history import apply_cold_history

def test_control_clone_and_prefix_only_left_history():
 x=torch.randn(4,700,176,dtype=torch.float32); y,l=apply_cold_history(x,seed=42,epoch=1,batch_id=3,probability=0)
 assert torch.equal(x,y) and y.data_ptr()!=x.data_ptr() and torch.equal(l,torch.full((4,),700,dtype=torch.int64))
 a,la=apply_cold_history(x,seed=42,epoch=1,batch_id=3); b,lb=apply_cold_history(x,seed=42,epoch=1,batch_id=3)
 assert torch.equal(a,b) and torch.equal(la,lb) and torch.equal(a[:,-1],x[:,-1])
 for row,n in enumerate(la.tolist()): assert torch.equal(a[row,700-n:],x[row,700-n:]) and not bool(a[row,:700-n].any())

@pytest.mark.parametrize('value',[torch.zeros(1,699,176),torch.zeros(1,700,176,dtype=torch.float64),torch.full((1,700,176),float('nan'))])
def test_rejects_bad_shape_dtype_and_finite(value):
 with pytest.raises(ValueError): apply_cold_history(value,seed=1,epoch=1,batch_id=1)
