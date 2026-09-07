"""No-NWB v3 closure, winner, cost and actual-ordinal contracts."""
from __future__ import annotations
import importlib.util,sys,json
from pathlib import Path
import numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mc_maze.t4_estimator_b_v3 as b

def s(name='s',a_shift=0):
 q=np.asarray(([0,1,2,3,4]*6)+([0,1,2,3,4]*4));th=np.asarray([b.v2.CANONICAL_DIRECTIONS_RAD[int(x)] for x in q]);c=np.rint(np.vstack([8+(2+a_shift)*np.cos(th),7-np.sin(th),5+.5*np.cos(th)])).astype(np.int64)
 return b.Session(name,c,np.ones(50),q,np.arange(50),name+'hash')
def row(r=.97,d=.03,ops=5,state=48):return {'ratio':r,'reliability':{'defined':True,'delta':d},'rank_increase':False,'nonconvergence_increase':False,'invalid_increase':False,'cost':{'calibration_ops_proxy':ops,'persistent_descriptor_state_bytes':state,'temporary_workspace_bytes':1}}
def result(r=.97,ops=5,state=48):return {'gate':{'pass':True,'mean_ratio':r},'rows':[row(r,.03,ops,state)]}
def test_winner_no_winner_tie_and_aggregate_cost():
 rs={'eb_ridge':result(.97,10,48),'second_harmonic':result(.96,100,48),'poisson_irls':{'gate':{'pass':False},'rows':[]}}
 assert b.winner(rs)['winner']=='second_harmonic';assert b.winner({k:{'gate':{'pass':False},'rows':[]} for k in b.CANDIDATES})['status']=='no_winner_no_gpu'
 tie={'eb_ridge':result(.96,10,48),'second_harmonic':result(.96,10,48),'poisson_irls':{'gate':{'pass':False},'rows':[]}}
 assert b.winner(tie)['status']=='no_winner_tie_after_frozen_breaks_no_gpu'; assert b.aggregate_cost(rs['eb_ridge']['rows'])['persistent_descriptor_state_bytes_mean']==48
def test_constrained_covariance_and_actual_poisson_ops():
 p=b.eb_prior([s('a',2),s('b',3)]);z=np.concatenate([np.column_stack([b.v2.ordinary(x).t4[:,3],b.v2.ordinary(x).t4[:,0],b.v2.ordinary(x).t4[:,1]]) for x in [s('a',2),s('b',3)]])
 expected=(z-np.asarray([z[:,0].mean(),0,0])).T@(z-np.asarray([z[:,0].mean(),0,0]))/(len(z)-1); assert np.allclose(p['covariance_bac']-np.eye(3)*p['regularization'],expected)
 f=b.v2.fit_poisson(s());cost=b._ops_state(s(),f,'poisson_irls');assert cost['persistent_descriptor_state_bytes']==3*4*4 and cost['calibration_ops_proxy']>0
def test_fixed_inner_and_eb_fast_inner_no_reliability(monkeypatch):
 sessions=[s(str(i)) for i in range(26)]
 fixed={'kind':'not_applicable_fixed_config','chosen_config':{'candidate':'second_harmonic'},'all_candidate_objectives':[]}; assert fixed['kind']=='not_applicable_fixed_config'
 monkeypatch.setattr(b.v2,'reliability',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not call reliability')))
 out=b.choose_eb_inner(sessions);assert len(out['all_candidate_objectives'])==3 and all(x['defined_count']==26 for x in out['all_candidate_objectives'])
def runner():
 p=Path(__file__).resolve().parents[1]/'scripts/audit_t4_estimator_b_v3.py';sp=importlib.util.spec_from_file_location('b3runner',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def test_actual_ordinal_mismatch_and_source_closure(monkeypatch,tmp_path):
 m=runner();r=m.prelaunch();assert m.verify(r);r['source_map']['x']='y'
 with pytest.raises(ValueError,match='source-map'):m.verify(r)
 p=tmp_path/'x.nwb';p.touch(); expected={'source_sha256':'ok','chronology_receipt':{'ordinals':list(range(50)),'start_times':list(range(50)),'stop_times':list(range(1,51)),'snapped_direction_indices':[0]*50}}
 monkeypatch.setattr(m,'sha',lambda _: 'ok');monkeypatch.setattr(m,'list_datamodule_rewarded_trials',lambda *a,**k:[{'trial_index':i+1,'start_time':i,'stop_time':i+1,'target_dir':0} for i in range(50)])
 with pytest.raises(ValueError,match='actual trial receipt mismatch'):m.load_checked('x',p,expected)
