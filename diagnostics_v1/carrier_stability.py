#!/usr/bin/env python3
"""CPU-only carrier construction audit; no decoder training or target-BP scoring.

Each surface binds a current production estimator, recomputes its carrier on
50/75/100-percent deterministic support subsets (seeds 101..110), and reports
Frobenius relative error/cosine versus the full-support carrier.  A task is
reported BLOCKED if its frozen inputs cannot reproduce that estimator exactly.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'btransform_unified_v1/src')]
SEEDS=tuple(range(101,111)); FRACS=(.50,.75,1.0)

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def subset(n:int, frac:float, seed:int)->np.ndarray:
 k=n if frac==1 else max(1,int(np.ceil(n*frac)))
 return np.arange(n,dtype=np.int64) if k==n else np.sort(np.random.default_rng(seed).choice(n,k,replace=False))
def comparison(full:np.ndarray, got:np.ndarray)->dict[str,float]:
 a=np.asarray(full,dtype=np.float64).ravel(); b=np.asarray(got,dtype=np.float64).ravel()
 den=float(np.linalg.norm(a)); return {'relative_frobenius':float(np.linalg.norm(b-a)/den) if den else float('nan'),'cosine':float(np.dot(a,b)/(den*np.linalg.norm(b))) if den and np.linalg.norm(b) else float('nan')}
def design_stats(theta:np.ndarray)->dict[str,Any]:
 x=np.stack([np.ones(len(theta)),np.cos(theta),np.sin(theta)],axis=1)
 return {'n':int(len(theta)),'rank':int(np.linalg.matrix_rank(x)),'condition':float(np.linalg.cond(x)) if np.linalg.matrix_rank(x)==3 else None}
def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
 out={}
 for frac in FRACS:
  z=[x for x in rows if x['fraction']==frac]
  for key in ('relative_frobenius','cosine'):
   v=np.array([x[key] for x in z],float); out.setdefault(str(int(frac*100)),{})[key]={'mean':float(np.nanmean(v)),'min':float(np.nanmin(v)),'max':float(np.nanmax(v))}
 return out

def m2(dest:Path)->dict[str,Any]:
 """M2 E0 is resampleable from cached trialized activity; MOVE-T4 is not.
 The latter needs un-interpolated calib bins/angles, absent from frozen cache.
 """
 from tfpd_exploration.src.m2_dual_track_v1 import champion, plan
 cache=ROOT/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'
 normal=json.loads((cache/'move_t4_normalizer.json').read_text())
 mean=np.asarray(normal['mean'],np.float32); std=np.asarray(normal['std'],np.float32)
 
 try:
  champion_obj=champion.load_frozen_champion(device='cpu')
 except Exception as exc:
  return {'status':'BLOCKED','binding':{'estimator':'B3S native_e0_and_u; MOVE-T4 fit_move_t4/t4_from_trial_sums','cache':str(cache),'normalizer_sha256':sha(cache/'move_t4_normalizer.json')},'blocker':repr(exc),'scope':'The SHA-pinned production B3S checkpoint is absent; cached E0/T artifacts are readable but cannot be called a fresh construction or stability result.'}
 sessions=[]; t0=time.perf_counter()
 for surface in ('source_train','ext4'):
  for p in sorted((cache/surface).iterdir()):
   if not p.is_dir():continue
   activity=np.load(p/'calib_activity.npy'); full_t=np.load(p/'T.npy')
   # Exact frozen T cannot be recomputed absent raw bins/angles; E0 materialization is exact.
   full_e0=champion.native_e0_and_u(champion_obj.encoder, __import__('torch').from_numpy(activity), champion.empty_contrast_side(full_t))[0].numpy()
   stored=__import__('torch').load(p/'e0_u.pt',map_location='cpu',weights_only=False)['E0'].numpy()
   rows=[]
   for frac in FRACS:
    for seed in SEEDS:
     ix=subset(len(activity),frac,seed); start=time.perf_counter()
     e0,_=champion.native_e0_and_u(champion_obj.encoder,__import__('torch').from_numpy(activity[ix]),champion.empty_contrast_side(full_t)); dt=time.perf_counter()-start
     row={'fraction':frac,'seed':seed,'n_support':int(len(ix)),'construct_seconds':dt,**comparison(full_e0,e0.numpy())}; rows.append(row)
   sessions.append({'surface':surface,'session':p.name,'n_trials':int(len(activity)),'stored_e0_equals_recomputed':bool(np.array_equal(stored,full_e0)),'identity_rows':rows,'identity_summary':summarize(rows),'carrier_full_sha256':sha(p/'T.npy')})
 return {'status':'PARTIAL','binding':{'estimator':'B3S native_e0_and_u; MOVE-T4 fit_move_t4/t4_from_trial_sums','cache':str(cache),'normalizer_sha256':sha(cache/'move_t4_normalizer.json'),'checkpoint_sha256':champion.CHAMPION_CKPT_SHA256},'scope':'E0 construction stability completed; exact MOVE-T4 carrier stability blocked because cache omits un-interpolated calib_neural, calib_trial_change, and angles required by fit_move_t4. Frozen T.npy is audited but not re-fit.','elapsed_seconds':time.perf_counter()-t0,'sessions':sessions}

def m1(dest:Path)->dict[str,Any]:
 from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data,plan
 from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3
 from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
 allrows=[]; t0=time.perf_counter()
 for fold in range(len(plan.FOLD_TARGETS)):
  loaded=data.load_fold_scope(fold=fold); records=dict(loaded['sources']); records[loaded['target'].session]=loaded['target']
  basis=syn3.fit_source_nmf(np.concatenate([rectify.relu_nonnegative_projection(r.emg) for r in loaded['sources'].values()]))
  def fit(record, ids):
   em=np.isin(record.emg_trial_ids,ids); rm=np.isin(record.rate_trial_ids,ids)
   scores=syn3.project_basis(rectify.relu_nonnegative_projection(record.emg[em]),basis)
   w,b=syn3.fit_all_units(scores,record.rates[rm]); return syn3.carrier_from_encoding(w,b)
  full={name:fit(r,np.arange(plan.SUPPORT_TRIALS)) for name,r in records.items()}
  norm_mean,norm_scale=syn3.source_normalizer([full[n] for n in loaded['sources']])
  per=[]
  for name,r in records.items():
   rows=[]
   for frac in FRACS:
    for seed in SEEDS:
     ix=subset(plan.SUPPORT_TRIALS,frac,seed); st=time.perf_counter(); got=syn3.normalize_carriers(fit(r,ix),norm_mean,norm_scale); dt=time.perf_counter()-st
     rows.append({'fraction':frac,'seed':seed,'n_support':int(len(ix)),'construct_seconds':dt,**comparison(syn3.normalize_carriers(full[name],norm_mean,norm_scale),got)})
   per.append({'session':name,'role':'source' if name in loaded['sources'] else 'heldout_support_only','rows':rows,'summary':summarize(rows)})
  allrows.append({'fold':fold,'target':loaded['target'].session,'source_sessions':list(loaded['sources']),'basis':'source-only rank-3 NNMF; per-unit ridge -> [3 weights, intercept]','normalizer':'source-only','coverage':'NNLS activations are used by project_basis; unit ridge is unconstrained','sessions':per,'isolation':loaded['isolation']})
 return {'status':'COMPLETE','binding':{'estimator':'m1_emg_rsyn3_fold_local_v1 carrier_bank::_encode_session','support_trials':int(plan.SUPPORT_TRIALS),'folds':len(plan.FOLD_TARGETS)},'elapsed_seconds':time.perf_counter()-t0,'folds':allrows}

def d688(dest:Path)->dict[str,Any]:
 from sua_exploration.mc_maze import unit_side_features as usf
 manifest=json.loads((ROOT/'sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json').read_text())
 paths={s:ROOT/'sua_exploration/data/dandi_000688/sub-C'/f'{s}_behavior+ecephys.nwb' for split in ('train','val') for s in manifest['session_splits'][split]}
 # Treat first 30 rewarded trials as established T4 calibration pool; direct raw feature fit only.
 rows=[]; t0=time.perf_counter()
 for session,path in paths.items():
  if not path.is_file(): raise FileNotFoundError(path)
  full,meta=usf.compute_unit_side_features_uncached(path,feature_group='t4',pool_size=30,signal_view='sua')
  # API only supports prefixes, so deterministic subsets cannot faithfully route through it.
  # Use 50/75/100 contiguous prefix construction as explicit protocol deviation blocker.
  rr=[]
  for frac in FRACS:
   n=max(1,int(np.ceil(30*frac))); st=time.perf_counter(); got,submeta=usf.compute_unit_side_features_uncached(path,feature_group='t4',pool_size=n,signal_view='sua'); dt=time.perf_counter()-st
   rr.append({'fraction':frac,'seed':'chronological_prefix_only','n_support':n,'construct_seconds':dt,**comparison(full,got),'design':{'available_via_public_api':False}})
  rows.append({'session':session,'split':'train' if session in manifest['session_splits']['train'] else 'val','full_shape':list(full.shape),'full_metadata':dict(meta.__dict__),'rows':rr})
 return {'status':'PARTIAL','binding':{'estimator':'unit_side_features.compute_unit_side_features_uncached(feature_group=t4)','manifest_sha256':sha(ROOT/'sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json'),'sessions':{'train':27,'val':6,'test_excluded':6}},'scope':'strict 27-train/6-val only; chronological 50/75/100 prefixes are measured. The production public API lacks selected-trial IDs, hence seeds101..110 cannot be truthfully applied without a new estimator entrypoint.','elapsed_seconds':time.perf_counter()-t0,'sessions':rows}

def h1(dest:Path)->dict[str,Any]:
 from btransform_unified_v1 import adapters,h1_config
 # Materializer input has exactly M3, so only full construction can be bound.
 rows=[]; t0=time.perf_counter()
 for s in h1_config.H1_ALL_SESSIONS:
  activity,carrier=adapters._h1_payload_arrays(s); st=time.perf_counter(); e0,hc=adapters._h1_materializer().materialize_bank(__import__('torch').from_numpy(activity).unsqueeze(0),__import__('torch').from_numpy(carrier).unsqueeze(0)); dt=time.perf_counter()-st
  rows.append({'session':s,'activity_shape':list(activity.shape),'carrier_shape':list(hc.shape),'full_construct_seconds':dt,'e0_sha256':hashlib.sha256(np.ascontiguousarray(e0.numpy()).tobytes()).hexdigest(),'carrier_sha256':hashlib.sha256(np.ascontiguousarray(hc.numpy()).tobytes()).hexdigest()})
 return {'status':'BLOCKED','binding':{'estimator':'C2-CAL-1 B2 fit_frozen_carrier/fit_deployment_carrier then C2 materializer','payload':str(h1_config.C2_M3_PAYLOAD_PATH),'payload_sha256':sha(h1_config.C2_M3_PAYLOAD_PATH),'materializer_checkpoint':str(h1_config.C2_CKPT_PATH),'materializer_checkpoint_sha256':sha(h1_config.C2_CKPT_PATH)},'blocker':'The frozen payload has exactly 3 trialized activity trials/session and its carrier is already materialized. Recomputing 50/75-percent seeded support carrier subsets needs the held-in calibration records plus B2 plan; current H1 production bank code explicitly exposes only M3 payload. Also current B2 source sets q=12/lambda=10, which conflicts with an older q=16/lambda=100 document.', 'elapsed_seconds':time.perf_counter()-t0,'full_m3_construction':rows}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=('m2','m1','d688','h1','all'),default='all');ap.add_argument('--dest',type=Path,default=ROOT/'diagnostics_v1/results/carrier_stability_v1'); args=ap.parse_args(); args.dest.mkdir(parents=True,exist_ok=True)
 os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
 result={'schema':'carrier_stability_construction_audit_v1','utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','affinity':sorted(os.sched_getaffinity(0)),'threads':2,'subsample_protocol':{'fractions':[50,75,100],'seeds':list(SEEDS),'metrics':['relative Frobenius norm vs full carrier','flattened cosine vs full carrier'],'100_percent':'one exact full-support construction per session; repeated only structurally for uniform rows where applicable'},'tasks':{}}
 funcs={'m2':m2,'m1':m1,'d688':d688,'h1':h1}
 for name,fn in funcs.items():
  if args.task not in ('all',name):continue
  try: result['tasks'][name]=fn(args.dest)
  except Exception as e: result['tasks'][name]={'status':'ERROR','error':repr(e),'traceback':traceback.format_exc()}
 (args.dest/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+'\n')
 print(args.dest/'report.json')
if __name__=='__main__':main()
