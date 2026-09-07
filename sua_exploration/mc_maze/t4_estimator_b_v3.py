"""Experiment-B v3 selection mechanics, append-only and CPU/source scoped.

v3 imports v2's direction-fair candidate fitters but replaces covariance, inner-selection,
cost accounting, and winner semantics.  A CPU winner can only request a new prelaunch; it can
never launch a GPU.
"""
from __future__ import annotations
import math,time
from typing import Mapping,Sequence
import numpy as np
from mc_maze import t4_estimator_b_v2 as v2

N_SOURCE=v2.N_SOURCE; CANDIDATES=v2.CANDIDATES; EB_GRID=v2.EB_GRID
Session=v2.Session
DESCRIPTOR_FLOAT_BYTES=4; DESCRIPTOR_FLOAT_DTYPE="float32"; OPS_VERSION="analytic_calibration_ops_v1"

def eb_prior(training:Sequence[v2.Session]):
 rows=[]
 for s in training:
  f=v2.ordinary(s); ok=np.isfinite(f.t4).all(1); rows.append(np.column_stack([f.t4[ok,3],f.t4[ok,0],f.t4[ok,1]]))
 z=np.concatenate(rows) if rows else np.empty((0,3))
 if len(z)<2: raise ValueError('no valid outer-training units')
 mean=np.asarray([float(z[:,0].mean()),0.,0.])
 # Deliberately not np.cov: a/c are centred at frozen zero, not their empirical sample means.
 cov=((z-mean).T@(z-mean))/float(len(z)-1); reg=max(float(np.trace(cov)/3),1e-6)*1e-5
 return {'mean_bac':mean,'covariance_bac':cov+np.eye(3)*reg,'regularization':reg,'source_names':[s.name for s in training],'source_hashes':[s.source_sha256 for s in training],'covariance_center':'frozen_[mu_b,0,0]'}

def _fit(session,train,candidate,config,counts=None):
 if candidate=='eb_ridge': return v2.fit_eb(session,eb_prior(train),float(config['lambda']),counts)
 if candidate=='second_harmonic': return v2.fit_2h(session,counts)
 if candidate=='poisson_irls': return v2.fit_poisson(session,counts)
 raise ValueError(candidate)

def _ops_state(session,fit,candidate):
 n=session.counts.shape[0]; _,x2,_,_,_,_,_=v2._aggregate(session); d=x2.shape[0]
 state=n*4*DESCRIPTOR_FLOAT_BYTES
 if candidate=='eb_ridge': ops=n*(d*18+54); tmp=n*(3*3+3)*8
 elif candidate=='second_harmonic': ops=n*(d*25+125); tmp=n*(5*5+5)*8
 else:
  records=fit.metadata.get('unit_records',[]); actual=sum(int(r['iterations']) for r in records); ops=actual*(d*18+27); tmp=n*(d*4+18)*8
 return {'ops_version':OPS_VERSION,'calibration_ops_proxy':int(ops),'persistent_descriptor_state_bytes':int(state),'persistent_descriptor_dtype':DESCRIPTOR_FLOAT_DTYPE,'temporary_workspace_bytes':int(tmp)}

def fast_ratio(session,train,config):
 """EB inner selection only: no 8-seed reliability; prospective deviance only."""
 f=_fit(session,train,'eb_ridge',config); o=v2.ordinary(session); od,cd=v2.deviance(session,o),v2.deviance(session,f)
 return None if od is None or cd is None or od<=0 else cd/od

def choose_eb_inner(train):
 table=[]
 for lam in EB_GRID:
  start=time.perf_counter(); vals=[]
  for i,held in enumerate(train): vals.append(fast_ratio(held,list(train[:i])+list(train[i+1:]),{'candidate':'eb_ridge','lambda':lam}))
  finite=[x for x in vals if x is not None and np.isfinite(x)]
  table.append({'config':{'candidate':'eb_ridge','lambda':lam},'per_inner_fold_ratio':vals,'defined_count':len(finite),'equal_session_mean_ratio':float(np.mean(finite)) if len(finite)==len(train) else None,'wall_clock_s':time.perf_counter()-start,'reliability_not_run':True})
 chosen=min(table,key=lambda x:(math.inf if x['equal_session_mean_ratio'] is None else x['equal_session_mean_ratio'],x['config']['lambda']))
 return {'kind':'nested_EB_prospective_deviance_only','chosen_config':chosen['config'],'all_candidate_objectives':table,'tie_break':'lower_equal_session_ratio_then_lower_lambda'}

def evaluate_outer(session,train,candidate,config):
 cost={}; s=time.perf_counter(); ordinary=v2.ordinary(session); cost['ordinary_wall_clock_s']=time.perf_counter()-s
 s=time.perf_counter(); fit=_fit(session,train,candidate,config); cost['candidate_wall_clock_s']=time.perf_counter()-s
 s=time.perf_counter(); rel=v2.reliability(session,lambda c:_fit(session,train,candidate,config,c)); cost['reliability_wall_clock_s']=time.perf_counter()-s
 od,cd=v2.deviance(session,ordinary),v2.deviance(session,fit); ratio=None if od is None or cd is None or od<=0 else cd/od
 cost.update(_ops_state(session,fit,candidate)); records=fit.metadata.get('unit_records',[]); it=[r['iterations'] for r in records]
 return {'session':session.name,'outer_train_names':[x.name for x in train],'outer_train_hashes':[x.source_sha256 for x in train],'config':dict(config),'ratio':ratio,'reliability':rel,'rank_increase':fit.rank_deficient>ordinary.rank_deficient,'nonconvergence_increase':fit.nonconverged>ordinary.nonconverged,'invalid_increase':fit.invalid>ordinary.invalid,'cost':cost,'poisson_iterations':None if not records else {'mean':float(np.mean(it)),'max':max(it),'q50':float(np.quantile(it,.5)),'q90':float(np.quantile(it,.9)),'failure_reasons':[r['failure_reason'] for r in records if r['failure_reason']]},'candidate_metadata':fit.metadata}

def gate(rows): return v2.gate(rows)
def run_candidate(sessions,candidate):
 if len(sessions)!=N_SOURCE or len({s.name for s in sessions})!=N_SOURCE:raise ValueError('requires exact 27 sources')
 rows=[]; failures=0
 for i,held in enumerate(sessions):
  train=list(sessions[:i])+list(sessions[i+1:])
  if candidate=='eb_ridge':
   start=time.perf_counter(); selection=choose_eb_inner(train); inner_elapsed=time.perf_counter()-start
  else:
   selection={'kind':'not_applicable_fixed_config','chosen_config':{'candidate':candidate},'all_candidate_objectives':[],'tie_break':None}; inner_elapsed=0.0
  row=evaluate_outer(held,train,candidate,selection['chosen_config']);row['inner_selection']=selection;row['cost']['inner_selection_wall_clock_s']=inner_elapsed; rows.append(row)
  good=row['ratio'] is not None and row['ratio']<=1 and row['reliability']['defined'] and row['reliability']['delta']>=0;failures+=int(not good)
  if failures>7:return {'candidate':candidate,'rows':rows,'gate':{'pass':False,'reason':'fail_fast_joint_20_of_27_impossible','joint_failures':failures,'remaining_not_run':N_SOURCE-len(rows)}}
 return {'candidate':candidate,'rows':rows,'gate':gate(rows)}

def aggregate_cost(rows):
 if not rows:return {'calibration_ops_proxy_mean':None,'persistent_descriptor_state_bytes_mean':None,'temporary_workspace_bytes_mean':None}
 return {'calibration_ops_proxy_mean':float(np.mean([r['cost']['calibration_ops_proxy'] for r in rows])),'persistent_descriptor_state_bytes_mean':float(np.mean([r['cost']['persistent_descriptor_state_bytes'] for r in rows])),'temporary_workspace_bytes_mean':float(np.mean([r['cost']['temporary_workspace_bytes'] for r in rows]))}
def winner(results:Mapping[str,Mapping[str,object]]):
 eligible=[]
 for name in CANDIDATES:
  r=results[name]
  if r['gate'].get('pass'):
   c=aggregate_cost(r['rows']); eligible.append((float(r['gate']['mean_ratio']),c['calibration_ops_proxy_mean'],c['persistent_descriptor_state_bytes_mean'],name))
 if not eligible:return {'status':'no_winner_no_gpu','winner':None,'eligible_candidates':[]}
 eligible.sort()
 if len(eligible)>1 and eligible[0][:3]==eligible[1][:3]:return {'status':'no_winner_tie_after_frozen_breaks_no_gpu','winner':None,'eligible_candidates':[x[3] for x in eligible]}
 return {'status':'unique_cpu_winner_requires_new_gpu_prelaunch','winner':eligible[0][3],'eligible_candidates':[x[3] for x in eligible],'tie_break':{'mean_ratio':eligible[0][0],'calibration_ops_proxy':eligible[0][1],'persistent_descriptor_state_bytes':eligible[0][2]}}
