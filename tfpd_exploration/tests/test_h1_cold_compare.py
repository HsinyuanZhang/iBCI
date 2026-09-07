import json
import pytest, torch
from tfpd_exploration.src.h1_family_v1 import cold_compare as c

def test_clone_cells_are_strictly_identical_within_arm_and_independent():
 bases={'FLAT':torch.nn.Linear(2,2),'ROUTE':torch.nn.Linear(2,2)}; cells=c.clone_cells(bases)
 assert set(cells)==set(c.CELLS)
 assert c.state_digest(cells['FLAT_CONTROL'].state_dict())==c.state_digest(cells['FLAT_PREFIX'].state_dict())
 assert cells['FLAT_CONTROL'] is not cells['FLAT_PREFIX']

def test_authorization_binds_protocol_code_inputs_output_and_smoke(tmp_path):
 output=tmp_path/'out'; smoke=tmp_path/'smoke.json'; auth=tmp_path/'auth.json'; code={'cold_compare.py':'x'}; bindings={'formal':'y'}
 smoke.write_text(json.dumps({'status':'PASS_32_UPDATE_RESOURCE_SMOKE','bindings':bindings}))
 body={'protocol':c.PROTOCOL,'protocol_sha256':c.digest(c.PROTOCOL),'code':code,'bindings':bindings,'output':str(output)};auth.write_text(json.dumps(body))
 assert c.require_authorization(auth,code=code,bindings=bindings,output=output,smoke=smoke)==body
 body['bindings']={};auth.write_text(json.dumps(body))
 with pytest.raises(RuntimeError): c.require_authorization(auth,code=code,bindings=bindings,output=output,smoke=smoke)

def test_runner_cannot_launch_without_review_go(monkeypatch):
 monkeypatch.delenv('H1_COLD_HISTORY_PROSPECTIVE_GO',raising=False)
 with pytest.raises(RuntimeError,match='explicit reviewed GO'): c.run()

class _M(torch.nn.Module):
 def __init__(self): super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.));self.calls=0
 def forward_last(self,x,bank,dropout_keep=None): self.calls+=1;return x[:,-1,:7]*self.w
class _O:
 def __init__(self,m):self.m=m;self.steps=0
 def zero_grad(self,set_to_none=True):self.m.w.grad=None
 def step(self):self.steps+=1
class _E:
 def __init__(self):self.count=0
 def update_after_step(self,m):self.count+=1
def test_train_step_micro8_weighted_for_variable_effective_batch():
 x=torch.ones(10,700,176);target=torch.ones(10,7);cells={k:_M() for k in c.CELLS};opts={k:_O(v) for k,v in cells.items()};emas={k:_E() for k in cells};keep=torch.ones(10,176,dtype=torch.bool)
 row=c.train_step(cells,opts,emas,x,target,object(),keep,epoch=1,batch_id=2)
 assert row['rows']==10 and set(row['loss'])==set(c.CELLS)
 assert all(m.calls==2 for m in cells.values()) and all(o.steps==1 for o in opts.values()) and all(e.count==1 for e in emas.values())
 assert 0<=row['cold_rows']<=10
