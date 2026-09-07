"""No-NWB unit tests for the DANDI 688 RIFT runner's padding contract."""
import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; PATH=ROOT/'scripts/rift_v1/dandi688_train.py'
spec=importlib.util.spec_from_file_location('dandi688_rift_runner',PATH); runner=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(runner)

def test_variable_units_are_right_padded_and_masked() -> None:
 r={'neural':np.zeros((80,6),np.float32),'behavior':np.zeros((80,2),np.float32),'starts':np.asarray([0,1]),'E0':np.ones((6,64),np.float32),'carrier':np.ones((6,4),np.float32),'unit_mask':np.asarray([1,1,1,1,0,0],bool)}
 x,y,v=runner.windows(r,r['starts'])
 assert x.shape==(2,50,6) and not r['unit_mask'][-2:].any()

def test_window_targets_use_last_bin() -> None:
 r={'neural':np.arange(80*3,dtype=np.float32).reshape(80,3),'behavior':np.arange(160,dtype=np.float32).reshape(80,2)}
 x,y,v=runner.windows(r,np.asarray([0,5],np.int64))
 assert x.shape==(2,50,3) and v.dtype==np.bool_ and v.all()
 assert np.array_equal(y[0],r['behavior'][49]) and np.array_equal(y[1],r['behavior'][54])

import json
import torch
import pytest


def _contract() -> dict:
    return {'contract': 'same'}

def test_resume_state_rejects_smoke_complete_or_wrong_epoch(tmp_path: Path) -> None:
    contract_sha=runner.obj_hash(_contract())
    base={'schema':runner.SCHEMA+'_checkpoint','status':'FORMAL_PARTIAL','smoke':False,'epochs':runner.EPOCHS,'contract_sha256':contract_sha,'epoch_zero_based':3,'next_epoch_zero_based':4,'global_step':4*28076,'raw_state_dict':{},'optimizer':{},'rng':{}}
    runner.validate_resume(base,contract_sha,tmp_path,28076)
    for key,value in [('smoke',True),('epoch_zero_based',11),('contract_sha256','wrong')]:
        bad={**base,key:value}
        if key=='epoch_zero_based': bad['next_epoch_zero_based']=12
        with pytest.raises(RuntimeError): runner.validate_resume(bad,contract_sha,tmp_path,28076)
    (tmp_path/'train_receipt.json').write_text('{}')
    with pytest.raises(RuntimeError,match='already complete'): runner.validate_resume(base,contract_sha,tmp_path,28076)


def test_score_requires_formal_average_and_persists_all_validation_arrays(tmp_path: Path) -> None:
    cs=runner.obj_hash(_contract())
    receipt={'schema':runner.SCHEMA+'_train_receipt','status':'TRAIN_COMPLETED','epochs':runner.EPOCHS,'updates_per_epoch':28076,'contract_sha256':cs,'formal_test_used':False}
    (tmp_path/'train_receipt.json').write_text(json.dumps(receipt))
    torch.save({'schema':runner.SCHEMA+'_average','status':'FORMAL','average_epochs_zero_based':list(runner.AVG),'contract_sha256':cs,'state_dict':{}},tmp_path/'average_e8_e11.pt')
    rows={}
    for i in range(6):
        rows[f'v{i}']={'split':'val','neural':np.zeros((52,2),np.float32),'behavior':(np.arange(52,dtype=np.float32)[:,None]+i).repeat(2,axis=1),'starts':np.asarray([0,1],np.int64),'e0':np.zeros((2,50),np.float32),'carrier':np.zeros((2,4),np.float32),'mask':np.ones(2,bool)}
    class Model:
        def load_state_dict(self, state): assert state=={}
        def eval(self): return self
        def __call__(self,x,b,**kw): return torch.ones((len(x),2),device=x.device)
    runner.score(tmp_path,rows,Model(),torch.device('cpu'),cs)
    result=json.loads((tmp_path/'score_receipt.json').read_text())
    assert result['status']=='SCORED' and len(result['sessions'])==6 and np.isfinite(result['equal_session_mean_mse']) and np.isfinite(result['equal_session_mean_r2']) and np.isfinite(result['pooled_r2'])
    saved=np.load(tmp_path/'scores'/'v5.npz')
    assert saved['prediction'].shape==(2,2) and np.array_equal(saved['query_start'],[0,1])
