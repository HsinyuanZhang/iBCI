"""No-NWB v4 ordinal-semantics reconciliation contracts."""
from __future__ import annotations
import importlib.util,sys,json
from pathlib import Path
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def runner():
 p=Path(__file__).resolve().parents[1]/'scripts/audit_t4_estimator_b_v4.py';sp=importlib.util.spec_from_file_location('b4runner',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def expected():return {'source_sha256':'ok','chronology_receipt':{'ordinals':list(range(50)),'start_times':list(range(50)),'stop_times':list(range(1,51)),'snapped_direction_indices':[0]*50}}
def test_reconciles_raw_indices_but_records_both(monkeypatch,tmp_path):
 m=runner();p=tmp_path/'x.nwb';p.touch(); monkeypatch.setattr(m,'sha',lambda _: 'ok'); raw=[0,1,2,3,4]+list(range(7,52))
 monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:[{'trial_index':raw[i],'start_time':i,'stop_time':i+1,'target_dir':0} for i in range(50)]); monkeypatch.setattr(m,'pool_trial_count_matrix_from_receipt',lambda *a,**k:np.ones((2,50),dtype=np.int64));e=expected();e['chronology_receipt']['snapped_direction_indices']=m.canonical_direction_indices([0]*50).tolist()
 s,r=m.load_checked('x',p,e);assert r['historical_selected_rewarded_ordinals'].tolist()==list(range(50));assert r['actual_raw_nwb_trial_indices'][5]==7 and s.trial_ordinals[5]==5
def test_identity_or_ordinal_semantics_mismatch_fail_closed(monkeypatch,tmp_path):
 m=runner();p=tmp_path/'x.nwb';p.touch();monkeypatch.setattr(m,'sha',lambda _: 'ok'); monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:[{'trial_index':i,'start_time':i,'stop_time':i+1,'target_dir':0} for i in range(50)])
 bad=expected();bad['chronology_receipt']['ordinals'][1]=99
 with pytest.raises(ValueError,match='ordinal semantics'):m.load_checked('x',p,bad)
 bad=expected();bad['chronology_receipt']['start_times'][3]=99
 with pytest.raises(ValueError,match='identity mismatch'):m.load_checked('x',p,bad)
def test_v4_source_closure_and_scope():
 m=runner();r=m.prelaunch();assert r['no_gpu'] and r['no_development_opened'] and r['no_formal_opened'] and m.verify(r);r['source_map']['x']='y'
 with pytest.raises(ValueError,match='source-map'):m.verify(r)
