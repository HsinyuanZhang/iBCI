#!/usr/bin/env python3
"""Frozen CPU-only M=4 empirical-Bayes shrinkage audit; no GPU/SPINT path."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import audit_h1_m4_population_decoder_carrier_date_lodo as raw

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL="h1_m4_empirical_bayes_confidence_carrier_date_lodo_cpu_v1"
PROTOCOL_PATH=ROOT/'sua_exploration/docs/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_SHRINKAGE_FROZEN_PROTOCOL.md'
RAW_RECEIPT=ROOT/'sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json'
NULL_N=31; SEED=20260807; MIN_DATES=4; THRESH=.5; EPS=1e-12
class AuditError(ValueError): pass
@dataclass(frozen=True)
class Frozen:
 outer:str; source:tuple[str,...]; mean:np.ndarray; scale:np.ndarray; pcs:np.ndarray; q:int; lam:float; U:np.ndarray; mu:np.ndarray; tau2:float; raw_plan_sha:str

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()
def _metric(y,p,m):
 s=float(np.square(y-p).sum()); t=float(np.square(y-m[None,:]).sum())
 return None if not np.isfinite(s) or not np.isfinite(t) or t<=EPS else {'r2':1-s/t,'sse':s,'tss':t}
def _weighted(rows):
 s=float(sum(x['sse'] for x in rows)); t=float(sum(x['tss'] for x in rows))
 return None if not np.isfinite(s) or not np.isfinite(t) or t<=EPS else {'r2':1-s/t,'sse':s,'tss':t}
def _rot(session,trial,n,j):
 if n<2: raise AuditError('rotation needs >=2 blocks')
 return 1+int.from_bytes(hashlib.sha256(f'{SEED}|{session}|{trial}|{j}'.encode()).digest()[:8],'big')%(n-1)
def _rowcos(a,b):
 d=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1); ok=d>EPS
 return None if not ok.any() else float(np.median((a[ok]*b[ok]).sum(1)/d[ok]))
def _support(record): return raw._support_query(record)
def _project(x,p): return ((x-p.mean)/p.scale)@p.pcs[:p.q].T
def _fit(record,p,indices=(0,1,2,3),override=None):
 s,_=_support(record); ts=[s[i] for i in indices]; x=np.concatenate([t.rates for t in ts]); y=np.concatenate([t.velocity if override is None else override[i] for i,t in zip(indices,ts)])
 z=_project(x,p); d=np.c_[np.ones(len(z)),z]; reg=np.eye(d.shape[1])*p.lam; reg[0,0]=0.; A=d.T@d+reg; beta=np.linalg.solve(A,d.T@y)
 rss=np.square(y-d@beta).sum(0); hat=float(np.trace(d@np.linalg.solve(A,d.T))); den=len(d)-hat
 if not np.isfinite(den) or den<=EPS: raise AuditError('ridge residual covariance degrees of freedom undefined')
 sig=rss/den; G=np.linalg.solve(A,d.T@d)@np.linalg.inv(A); G=(G+G.T)/2
 return beta,G,sig
def _rawrows(beta,p): return (p.pcs[:p.q].T@beta[1:])/p.scale[:,None]
def _shrink(beta,G,sig,p):
 R=_rawrows(beta,p); E=R@p.U; P=p.pcs[:p.q].T; f=((P@G[1:,1:])*P).sum(1)/np.square(p.scale)
 cov4=p.U.T@np.diag(sig)@p.U; v=f*np.trace(cov4)/4
 if np.any(~np.isfinite(v)) or np.any(v<0) or not np.isfinite(p.tau2) or p.tau2<=EPS: raise AuditError('analytic projected variance/prior undefined')
 w=p.tau2/(p.tau2+v); return p.mu[None,:]+w[:,None]*(E-p.mu[None,:]),w,v
def _predict_rows_metric(record,R,beta0,p):
 _,q=_support(record); m=np.concatenate([t.velocity for t in record.trials[:4]]).mean(0); rows=[]
 for t in q: rows.append(_metric(t.velocity,(t.rates-p.mean)@R+beta0,m))
 return None if any(x is None for x in rows) else _weighted(rows)
def _predict_metric(record,E,beta0,p): return _predict_rows_metric(record,E@p.U.T,beta0,p)
def _frozen_plan(records,outer,bound):
 b=bound[outer]['plan']; src=tuple(b['source_sessions']);
 if any(records[n].date==outer for n in src): raise AuditError('outer leak')
 x=np.concatenate([raw._support_rates(records[n]) for n in src]); mean=x.mean(0); scale=np.maximum(x.std(0),1e-6); _,_,pcs=np.linalg.svd((x-mean)/scale,full_matrices=False); q=int(b['q']); lam=float(b['lambda'])
 tmp=Frozen(outer,src,mean,scale,pcs[:16],q,lam,np.empty((7,4)),np.zeros(4),1.,b['plan_sha256'])
 Rs=np.concatenate([_rawrows(_fit(records[n],tmp)[0],tmp) for n in src]); _,_,v=np.linalg.svd(Rs,full_matrices=False); U=v[:4].T; Es=Rs@U; mu=Es.mean(0); tau2=float(np.square(Es-mu).sum()/(Es.shape[0]*4))
 return Frozen(outer,src,mean,scale,pcs[:16],q,lam,U,mu,tau2,b['plan_sha256'])
def _shuffle(E,record,p):
 n=len(E); seed=int.from_bytes(hashlib.sha256(f'{SEED}|row|{record.session_name}|{p.outer}'.encode()).digest()[:8],'big'); perm=np.random.default_rng(seed).permutation(n)
 if np.array_equal(perm,np.arange(n)): perm=np.roll(perm,1)
 return E[perm], hashlib.sha256(np.ascontiguousarray(perm).tobytes()).hexdigest()
def _record(record,p):
 beta,G,sig=_fit(record,p); R=_rawrows(beta,p); E,w,v=_shrink(beta,G,sig,p); correct=_predict_metric(record,E,beta[0],p); rawm=_predict_rows_metric(record,R,beta[0],p)
 ba,Ga,sa=_fit(record,p,(0,1)); bb,Gb,sb=_fit(record,p,(2,3)); EA,_,_=_shrink(ba,Ga,sa,p); EB,_,_=_shrink(bb,Gb,sb,p); cos=_rowcos(EA,EB); rawcos=_rowcos(_rawrows(ba,p),_rawrows(bb,p))
 ES,permsha=_shuffle(E,record,p); shuffled=_predict_metric(record,ES,beta[0],p); s,_=_support(record); null=[]
 for j in range(NULL_N):
  over={i:np.roll(t.velocity,_rot(record.session_name,t.trial_number,len(t.velocity),j),0) for i,t in enumerate(s)}; b0,g0,s0=_fit(record,p,override=over); e0,_,_=_shrink(b0,g0,s0,p); null.append(_predict_metric(record,e0,b0[0],p))
 ok=all(x is not None for x in [correct,rawm,shuffled]) and cos is not None and rawcos is not None and all(x is not None for x in null)
 return {'status':'defined' if ok else 'undefined','session_name':record.session_name,'correct':correct,'raw':rawm,'shuffle':shuffled,'shrunk_cosine':cos,'raw_cosine_fresh':rawcos,'shrink_weight_mean':float(w.mean()),'projected_variance_mean':float(v.mean()),'carrier_shape':[int(E.shape[0]),int(E.shape[1])],'carrier_sha256':hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest(),'row_shuffle_permutation_sha256':permsha,'null':null}
def _date(rows,p,bound):
 ok=bool(rows) and all(x['status']=='defined' for x in rows)
 def agg(key): return _weighted([x[key] for x in rows]) if ok else None
 c,r,s=agg('correct'),agg('raw'),agg('shuffle'); null=[] if ok else None
 if null is not None:
  for j in range(NULL_N): null.append(_weighted([x['null'][j] for x in rows])['r2'])
 sc=None if not ok else float(np.mean([x['shrunk_cosine'] for x in rows])); rc=None if not ok else float(np.mean([x['raw_cosine_fresh'] for x in rows])); q95=None if null is None else float(np.quantile(null,.95)); br=bound[p.outer]
 # strict binding to prior immutable M4 receipt
 if c is None or r is None or rc is None or abs(r['r2']-br['correct_later_query_variance_weighted_r2_relative_fit_side_mean'])>1e-10 or abs(rc-br['split_pair_coefficient_row_attachment_mean_median_cosine'])>1e-10: raise AuditError('raw comparator does not bind to immutable M4 receipt')
 return {'date':p.outer,'status':'defined','plan':{'raw_plan_sha256':p.raw_plan_sha,'source_sessions':list(p.source),'q':p.q,'lambda':p.lam,'prior_mu':p.mu.tolist(),'prior_tau2':p.tau2},'shrunk_r2':c['r2'],'raw_m4_r2_bound':br['correct_later_query_variance_weighted_r2_relative_fit_side_mean'],'raw_m4_r2_fresh':r['r2'],'null_q95':q95,'shrunk_minus_null_q95':c['r2']-q95,'shrunk_split_pair_cosine':sc,'raw_m4_split_pair_cosine_bound':br['split_pair_coefficient_row_attachment_mean_median_cosine'],'raw_m4_split_pair_cosine_fresh':rc,'row_shuffle_r2':s['r2'],'shrunk_minus_row_shuffle':c['r2']-s['r2'],'records':rows}
def run_audit(records,bound):
 if tuple(sorted({x.date for x in records.values()}))!=tuple(raw.m2.h1.H1_DATES): raise AuditError('six authority dates required')
 out=[]
 for d in raw.m2.h1.H1_DATES:
  try:
   p=_frozen_plan(records,d,bound); out.append(_date([_record(x,p) for x in records.values() if x.date==d],p,bound))
  except Exception as e: out.append({'date':d,'status':'undefined','reason':str(e)})
 keys={'positive':'shrunk_r2','null':'shrunk_minus_null_q95','stable':'shrunk_split_pair_cosine','improved_cos':'_cosdiff','raw':'_rawdiff','shuffle':'shrunk_minus_row_shuffle'}
 def val(row,k):
  if k=='_cosdiff': return row.get('shrunk_split_pair_cosine',-np.inf)-row.get('raw_m4_split_pair_cosine_bound',np.inf)
  if k=='_rawdiff': return row.get('shrunk_r2',-np.inf)-row.get('raw_m4_r2_bound',np.inf)
  return row.get(k,-np.inf)
 counts={n:sum(val(x,k)>=(THRESH if n=='stable' else 0) for x in out) for n,k in keys.items()}
 gate={'all_six_dates_defined':all(x['status']=='defined' for x in out),'shrunk_r2_positive_dates':counts['positive'],'shrunk_above_fresh_null_q95_dates':counts['null'],'shrunk_split_pair_cosine_at_least_050_dates':counts['stable'],'shrunk_cosine_above_raw_dates':counts['improved_cos'],'shrunk_r2_at_least_raw_dates':counts['raw'],'shrunk_r2_above_row_shuffle_dates':counts['shuffle'],'minimum_required_dates':MIN_DATES}; gate['pass']=gate['all_six_dates_defined'] and all(x>=MIN_DATES for x in counts.values())
 return {'schema':'h1_m4_empirical_bayes_confidence_carrier_date_lodo_cpu_v1','protocol':PROTOCOL,'status':'PASS_CPU_H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER__NO_GPU_AUTHORIZATION' if gate['pass'] else 'STOP_CPU_H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_FAILED__NO_GPU_AUTHORIZATION','scope':{'opened_data':'exactly 13 public held-in-calib NWBs','formal_heldout_opened':False,'minival_or_query_opened':False,'gpu_constructed':False,'target_backpropagation':False},'source_hashes':{'audit_script_sha256':sha(Path(__file__)),'protocol_sha256':sha(PROTOCOL_PATH),'raw_m4_receipt_sha256':sha(RAW_RECEIPT),'raw_m4_script_sha256':sha(Path(raw.__file__))},'input_nwb_sha256':{n:x.input_sha256 for n,x in records.items()},'design':{'carrier':'[N,4] empirical-Bayes shrunk population decoder row carrier','scoring':'R_C=E_C U^T; yhat=(x-source_mean)R_C+pooled_intercept','source_plan':'bound to immutable M4 receipt; no grid/sweep','null':'31 fresh within-support rotations; query correct','row_shuffle':'deterministic complete nonidentity row permutation'},'date_lodo':out,'cpu_gate':gate,'decision':{'gpu_authorized_by_this_receipt':False,'if_fail':'STOP permanently; do not alter prior, covariance, thresholds, or source plan.','m2_and_raw_m4_routes_unchanged':True}}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,default=ROOT/'SPINT-main/data/000954'); ap.add_argument('--output',type=Path,default=ROOT/'sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json'); a=ap.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,''): raise RuntimeError('CPU only requires CUDA unset')
 if a.output.exists(): raise FileExistsError(a.output)
 bound={x['date']:x for x in json.load(RAW_RECEIPT.open())['date_lodo']}; paths=raw.m2.h1.index_h1_heldin_calib(a.data_dir); o=run_audit({n:raw.m2.load_full_h1_record(p) for n,p in paths.items()},bound); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); a.output.chmod(0o444); print(json.dumps({'status':o['status'],'output':str(a.output)}))
if __name__=='__main__': main()
