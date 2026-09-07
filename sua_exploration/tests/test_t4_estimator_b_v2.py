"""Synthetic/no-NWB v2 contracts, including v1 NO-GO remediation boundaries."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mc_maze.t4_estimator_b_v2 as b

def sess(name='s',counts=None,directions=None):
 d=np.ones(50); q=np.asarray(([0,1,2,3,4]*6)+([0,1,2,3,4]*4),dtype=np.int64) if directions is None else directions
 theta=np.asarray([b.CANONICAL_DIRECTIONS_RAD[int(x)] for x in q]); rate=np.vstack([9+2*np.cos(theta)+.5*np.cos(2*theta),7-np.sin(theta)+.4*np.sin(2*theta),5+.8*np.cos(theta)])
 c=np.rint(rate).astype(np.int64) if counts is None else counts
 return b.Session(name,c,d,q,np.arange(50),f'hash-{name}')

def test_direction_row_fairness_replication_imbalance_invariance():
 a=sess('a')
 # Same per-direction rates, but support replication differs (10,5,5,5,5 vs 6 each).
 q=np.asarray(([0]*10)+([1]*5)+([2]*5)+([3]*5)+([4]*5)+([0,1,2,3,4]*4),dtype=np.int64); theta=np.asarray([b.CANONICAL_DIRECTIONS_RAD[int(x)] for x in q]); c=np.rint(np.vstack([9+2*np.cos(theta)+.5*np.cos(2*theta),7-np.sin(theta)+.4*np.sin(2*theta),5+.8*np.cos(theta)])).astype(np.int64); z=sess('z',c,q)
 prior=b.eb_prior([a,sess('p')])
 for f in (lambda s:b.ordinary(s),lambda s:b.fit_eb(s,prior,1.),lambda s:b.fit_2h(s),lambda s:b.fit_poisson(s)):
  assert np.allclose(f(a).t4,f(z).t4,atol=1e-7,equal_nan=True)
 assert b.fit_poisson(a).metadata['aggregate'].startswith('counts_and_exposure_by_direction')

def test_eb_zero_direction_mean_and_heldout_exclusion():
 a,p,h=sess('a'),sess('p'),sess('held')
 prior=b.eb_prior([a,p]); assert prior['mean_bac'][1:].tolist()==[0.,0.] and 'held' not in prior['source_names']
 f=b.fit_eb(h,prior,.25); assert f.metadata['prior']['source_names']==['a','p']

def test_2h_export_score_and_poisson_iteration_failure(monkeypatch):
 s=sess(); f=b.fit_2h(s); assert f.t4.shape[1]==4 and f.metadata['export_and_score']=='first_harmonic_only'; assert b.deviance(s,f) is not None
 def bad(*a,**k): raise np.linalg.LinAlgError('x')
 monkeypatch.setattr(np.linalg,'solve',bad); p=b.fit_poisson(s); assert p.nonconverged==s.counts.shape[0] and all(not x['converged'] for x in p.metadata['unit_records'])

def test_undefined_full27_gate_and_irreversible_failfast(monkeypatch):
 rows=[{'ratio':.97,'reliability':{'defined':True,'delta':.03},'rank_increase':False,'nonconvergence_increase':False,'invalid_increase':False} for _ in range(27)]; rows[2]['ratio']=None
 assert b.gate(rows)['reason']=='undefined_ratio_or_reliability'
 sessions=[sess(f's{i}') for i in range(27)]
 monkeypatch.setattr(b,'choose_inner',lambda tr,c:{'chosen_config':{'candidate':c},'all_candidate_objectives':[]})
 monkeypatch.setattr(b,'evaluate',lambda *a,**k:{'ratio':2.,'reliability':{'defined':True,'delta':-1.},'rank_increase':False,'nonconvergence_increase':False,'invalid_increase':False,'cost':{}})
 assert b.run_candidate(sessions,'second_harmonic')['gate']['reason']=='fail_fast_joint_20_of_27_impossible'

def runner():
 p=Path(__file__).resolve().parents[1]/'scripts/audit_t4_estimator_b_v2.py'; sp=importlib.util.spec_from_file_location('b2runner',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def test_receipt_closure_tamper_and_formal_dev_nonresolution(tmp_path):
 m=runner(); r=m.prelaunch(); assert r['no_development_opened'] and r['no_formal_opened'] and m.verify(r)
 r['source_map']['sua_exploration/mc_maze/t4_estimator_b_v2.py']='tamper'
 with pytest.raises(ValueError,match='source-map'):m.verify(r)
 base=tmp_path/'sub-C';base.mkdir(); names=[f'n{i}' for i in range(27)]
 for n in names:(base/f'{n}_behavior+ecephys.nwb').touch()
 out=m.resolve_sources({'source_sessions':names,'sealed_development_names':['dev'],'sealed_formal_names':['formal']},tmp_path);assert len(out)==27 and not (base/'dev_behavior+ecephys.nwb').exists() and not (base/'formal_behavior+ecephys.nwb').exists()

def test_source_sha_and_chronology_mismatch_fail_before_count_read(monkeypatch,tmp_path):
 m=runner(); p=tmp_path/'x.nwb';p.touch(); expected={'source_sha256':'expected','chronology_receipt':{'start_times':[0]*50,'stop_times':[1]*50,'snapped_direction_indices':[0]*50,'ordinals':list(range(50))}}
 monkeypatch.setattr(m,'sha',lambda x:'actual')
 with pytest.raises(ValueError,match='SHA mismatch'):m.load_checked('x',p,expected)
 monkeypatch.setattr(m,'sha',lambda x:'expected'); monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:[{'start_time':i,'stop_time':i+1,'target_dir':0} for i in range(50)])
 with pytest.raises(ValueError,match='chronology mismatch'):m.load_checked('x',p,expected)
