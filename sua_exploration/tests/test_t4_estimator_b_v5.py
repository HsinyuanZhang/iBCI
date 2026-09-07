"""v5 retains the v3 frozen contract while adding dual ordinal provenance."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def runner():
 p=Path(__file__).resolve().parents[1]/'scripts/audit_t4_estimator_b_v5.py';sp=importlib.util.spec_from_file_location('b5runner',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def exp():return {'source_sha256':'ok','chronology_receipt':{'ordinals':list(range(50)),'start_times':list(range(50)),'stop_times':list(range(1,51)),'snapped_direction_indices':[0]*50}}
def test_full_v3_contract_retained_and_source_closure():
 m=runner();r=m.prelaunch();assert len(r['source_sessions'])==27
 assert r['chronology']=={'trials':50,'support':[0,30],'score':[30,50],'actual_trial_index_required':True}
 assert r['winner']['eligible']=='gate.pass_only' and r['winner']['automatic_gpu'] is False
 assert r['cost']['actual_poisson_iterations_determine_ops'] and r['runtime_work_receipt']['EB_inner_prospective_fits']==2106
 assert 't4_cross_budget_features.py' in ''.join(r['source_map']) and m.verify(r);r['source_map']['x']='x'
 with pytest.raises(ValueError,match='source-map'):m.verify(r)
def test_raw_nonmonotonic_and_individual_identity_fields_fail(monkeypatch,tmp_path):
 m=runner();p=tmp_path/'x.nwb';p.touch();monkeypatch.setattr(m,'sha',lambda _: 'ok')
 def rows(raw=None,start_shift=0,stop_shift=0,target=0):return [{'trial_index':i if raw is None else raw[i],'start_time':i+start_shift,'stop_time':i+1+stop_shift,'target_dir':target} for i in range(50)]
 monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:rows(raw=[0]*50))
 with pytest.raises(ValueError,match='strictly increasing'):m.load_checked('x',p,exp())
 monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:rows(start_shift=1))
 with pytest.raises(ValueError,match='start mismatch'):m.load_checked('x',p,exp())
 monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:rows(stop_shift=1))
 with pytest.raises(ValueError,match='stop mismatch'):m.load_checked('x',p,exp())
 monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:rows(target=1))
 with pytest.raises(ValueError,match='direction mismatch'):m.load_checked('x',p,exp())
 bad=exp();bad['chronology_receipt']['ordinals'][2]=3
 monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:rows())
 with pytest.raises(ValueError,match='ordinal semantics'):m.load_checked('x',p,bad)
