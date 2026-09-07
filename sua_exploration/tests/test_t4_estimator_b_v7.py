"""No-NWB execution binding test for final v7 runner."""
from __future__ import annotations
import importlib.util,sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def runner():
 p=Path(__file__).resolve().parents[1]/'scripts/audit_t4_estimator_b_v7.py';sp=importlib.util.spec_from_file_location('b7runner',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def test_execute_embeds_v7_prelaunch_winner_scope_and_fresh_output(monkeypatch,tmp_path):
 m=runner(); r=m.prelaunch(); assert m.verify(r) and r['execution']['automatic_gpu'] is False
 fake_paths={n:tmp_path/f'{n}.nwb' for n in r['source_sessions']}; monkeypatch.setattr(m,'resolve_sources',lambda receipt,data:fake_paths)
 monkeypatch.setattr(m,'load_checked',lambda n,p,e:(type('S',(),{'name':n})(),{'name':n,'dual_ordinal':True}))
 monkeypatch.setattr(m,'run_candidate',lambda sessions,c:{'candidate':c,'rows':[],'gate':{'pass':False}})
 monkeypatch.setattr(m,'aggregate_cost',lambda rows:{'calibration_ops_proxy_mean':None})
 monkeypatch.setattr(m,'winner',lambda results:{'status':'no_winner_no_gpu','winner':None})
 out=tmp_path/'out'; payload=m.execute_source_only(tmp_path,out); saved=json.loads((out/'aggregate.json').read_text())
 assert payload['prelaunch']['schema']=='t4_estimator_b_v7_prelaunch_v1' and saved['prelaunch']['schema']=='t4_estimator_b_v7_prelaunch_v1'
 assert saved['no_gpu'] and saved['no_development_opened'] and saved['no_formal_opened'] and saved['winner']['status']=='no_winner_no_gpu'
 with pytest.raises(FileExistsError):m.execute_source_only(tmp_path,out)
def test_v7_source_map_contains_direct_v6_v5_v3_runners():
 m=runner();r=m.prelaunch();assert m.verify(r)
 for path in ('sua_exploration/scripts/audit_t4_estimator_b_v6.py','sua_exploration/scripts/audit_t4_estimator_b_v5.py','sua_exploration/scripts/audit_t4_estimator_b_v3.py'):
  assert path in r['source_map']
