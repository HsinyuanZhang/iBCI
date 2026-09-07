from __future__ import annotations
import importlib.util,sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def runner():
 p=Path(__file__).resolve().parents[1]/'scripts/audit_t4_estimator_b_v6.py';sp=importlib.util.spec_from_file_location('b6runner',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def test_complete_canonical_contract_and_direct_runner_closure():
 m=runner();r=m.prelaunch();assert m.verify(r);assert 'sua_exploration/scripts/audit_t4_estimator_b_v3.py' in r['source_map'];assert r['chronology']['score']==[30,50] and r['winner']['automatic_gpu'] is False
def test_all_material_receipt_drift_fails_closed():
 m=runner();r=m.prelaunch()
 for mutate in (lambda x:x['winner'].__setitem__('order',['bad']),lambda x:x['chronology'].__setitem__('score',[31,50]),lambda x:x['ordinal_provenance'].__setitem__('current_field','bad'),lambda x:x['source_map'].__setitem__('x','y')):
  bad=copy.deepcopy(r);mutate(bad)
  with pytest.raises(ValueError,match='canonical'):m.verify(bad)
