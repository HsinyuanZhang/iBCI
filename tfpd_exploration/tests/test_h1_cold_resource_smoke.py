from types import SimpleNamespace
import pytest, torch
from tfpd_exploration.src.h1_family_v1 import cold_resource_smoke as smoke

def test_gate_and_nonoverwrite_precede_formal_audit(tmp_path,monkeypatch):
 monkeypatch.delenv('H1_COLD_RESOURCE_SMOKE_GO',raising=False);monkeypatch.setattr(smoke,'sha',lambda p:pytest.fail('must not hash/audit before gate'))
 with pytest.raises(RuntimeError,match='explicit smoke GO'):smoke.preflight(tmp_path/'f',tmp_path/'o',arm='flat',device='cuda:0',physical_gpu=0)
 out=tmp_path/'o';out.mkdir();monkeypatch.setenv('H1_COLD_RESOURCE_SMOKE_GO','1');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 with pytest.raises(FileExistsError):smoke.preflight(tmp_path/'f',out,arm='flat',device='cuda:0',physical_gpu=0)

class E:
 def __init__(self):self.n_updates=0
 def update_after_step(self,m):self.n_updates+=1
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.));self.batch_calls=[]
 def forward_last(self,x,bank,dropout_keep=None):self.batch_calls.append(len(x));return x[:,-1,:7]*self.w
class O:
 def __init__(self,m):self.m=m;self.steps=0
 def zero_grad(self,set_to_none=True):self.m.w.grad=None
 def step(self):
  self.steps+=1
  with torch.no_grad():self.m.w.add_(self.m.w.grad,alpha=-1e-5)
def test_matched_train_step_updates_both_conditions_and_ema_counts():
 x=torch.ones(32,700,176);target=torch.ones(32,7);cells={k:M() for k in ('CONTROL','PREFIX')};opts={k:O(v) for k,v in cells.items()};emas={k:E() for k in cells};keep=torch.ones(32,176,dtype=torch.bool)
 losses,lengths=smoke._run_update(torch,cells,opts,emas,(x,target,object(),keep),epoch=1,batch_id=0)
 assert set(losses)=={'CONTROL','PREFIX'} and all(torch.isfinite(torch.tensor(v)) for v in losses.values()) and all(v.n_updates==1 for v in emas.values())
 assert lengths.shape==(32,) and torch.equal(x[:,-1],apply_last:=x[:,-1])
 assert all(m.batch_calls==[8,8,8,8] for m in cells.values())
 assert all(o.steps==1 for o in opts.values())

def test_ragged_batch_uses_true_micro_and_weighted_loss():
 x=torch.ones(10,700,176);x[8:,-1,:7]=2
 cells={k:M() for k in ('CONTROL','PREFIX')};opts={k:O(v) for k,v in cells.items()};emas={k:E() for k in cells}
 losses,_=smoke._run_update(torch,cells,opts,emas,(x,torch.zeros(10,7),object(),torch.ones(10,176,dtype=torch.bool)),epoch=1,batch_id=1)
 assert all(m.batch_calls==[8,2] for m in cells.values())
 assert all(o.steps==1 for o in opts.values())
 assert all(v==pytest.approx(1.6) for v in losses.values())

def test_budget_counts_both_conditions_in_one_sample():
 result=smoke.project_budget([1.]*28)
 assert result['training_2gpu_2epochs_4cells_50pct_margin']==2193
 assert result['total_conservative']==6693
 with pytest.raises(RuntimeError):smoke.project_budget([1.]*27)
 with pytest.raises(RuntimeError):smoke.project_budget([float('nan')]*28)

def test_ema_first_successful_update_copies_raw_state():
 m=torch.nn.Linear(2,2);ema=smoke._fresh_ema(m);ema.update_after_step(m)
 assert ema.n_updates==1 and all(torch.equal(v,m.state_dict()[k]) for k,v in ema.shadow.items())
