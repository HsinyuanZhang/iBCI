#!/usr/bin/env python3
"""Create sealed source-only [unit,4] carrier packs for the last-one protocol."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT.parent/'btransform_unified_v1/src'),str(ROOT/'scripts'),str(ROOT),str(ROOT.parent),str(HERE)]
from carrier_profile_v2 import m1_profiles, h1_profiles
from carrier_profile_v2 import m1_muscle_profile

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ah(x):
 a=np.ascontiguousarray(x);return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def m1_path(s): return ROOT.parent/'SPINT-main/data/000941/sub-MonkeyL-held-in-calib'/f'sub-MonkeyL-held-in-calib_{s}_behavior+ecephys.nwb'
def _jsonable(x):
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,dict): return {str(k):_jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [_jsonable(v) for v in x]
 return x
def atomic_npz(p, **kw):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp.npz');np.savez_compressed(t,**kw);t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('m1','h1'),required=True);p.add_argument('--profile-id',choices=('old','encoding','state','muscle'),required=True);p.add_argument('--stage',choices=('inner','outer'),required=True);p.add_argument('--surface',choices=('source','target'),default='source');p.add_argument('--source-pack',type=Path);p.add_argument('--source-gate',type=Path);p.add_argument('--dest',type=Path,required=True);a=p.parse_args()
 if a.dest.exists(): raise FileExistsError(f'refuse overwrite {a.dest}')
 if a.dataset=='m1':
  if a.profile_id=='muscle': cand='muscle_response16_svd4'; mod=m1_muscle_profile
  else: cand={'old':'old_rsyn3','encoding':'standardized_rsyn3','state':'soft_prototype4'}[a.profile_id]; mod=m1_profiles
  sources=('ses-20120924','ses-20120926') if a.stage=='inner' else ('ses-20120924','ses-20120926','ses-20120927')
  targets=() if a.surface=='source' else (('ses-20120927',) if a.stage=='inner' else ('ses-20120928',))
  seal=hashlib.sha256((a.stage+'|'+a.profile_id+'|'+'|'.join(sources)).encode()).hexdigest()
  arrays={};details={}; source_pack_sha=None; source_gate_sha=None
  if a.surface=='source':
   fit=mod.fit_source(sources,m1_path,normalization='global_rms') if a.profile_id=='muscle' else mod.fit_source_profile(cand,sources,m1_path)
   frozen=a.dest.with_suffix('.frozen.npz'); digest=(mod.save_frozen_fit if a.profile_id=='muscle' else mod.save_frozen_profile_fit)(fit,frozen,source_seal=seal)
   for name in sources:
    z,d=(mod.project(fit,m1_path(name)) if a.profile_id=='muscle' else mod.project_profile(fit,m1_path(name)));arrays[f'carrier/{name}']=z;details[name]=_jsonable(d)
  else:
   if a.source_pack is None or a.source_gate is None or not a.source_gate.is_file(): raise RuntimeError('M1 target requires --source-pack and explicit --source-gate')
   gate=json.loads(a.source_gate.read_text()); source_gate_sha=sha(a.source_gate)
   if gate.get('schema')!='carrier_last1_v2_paired_audit' or gate.get('status')!='PASSED' or gate.get('stage')!=a.stage or gate.get('dataset')!='m1': raise RuntimeError('source gate identity mismatch')
   with np.load(a.source_pack,allow_pickle=False) as src:
    smeta=json.loads(str(src['metadata'].item()))
   if (smeta.get('schema')!='carrier_last1_v2_pack' or smeta.get('dataset')!='m1' or smeta.get('profile_id')!=a.profile_id or smeta.get('stage')!=a.stage or smeta.get('surface')!='source' or tuple(smeta.get('sources',()))!=sources): raise RuntimeError('M1 source pack identity drift')
   frozen=Path(smeta.get('frozen_fit','')).resolve(); expected=smeta.get('frozen_fit_sha256')
   if not frozen.is_file() or expected!=sha(frozen): raise RuntimeError('M1 frozen fit binding drift')
   source_pack_sha=sha(a.source_pack)
   fit=(mod.load_frozen_fit if a.profile_id=='muscle' else mod.load_frozen_profile_fit)(frozen,source_seal=seal); digest=sha(frozen)
   for name in targets:
    if a.stage=='inner':
     z,d=(mod.project(fit,m1_path(name)) if a.profile_id=='muscle' else mod.project_profile(fit,m1_path(name)))
    else:
     z,d=mod.deploy_project_m10(fit,m1_path(name),source_seal=seal)
    arrays[f'carrier/{name}']=z;details[name]=_jsonable(d)
  meta={'schema':'carrier_last1_v2_pack','dataset':'m1','profile_id':a.profile_id,'candidate':cand,'stage':a.stage,'sources':list(sources),'targets':list(targets),'carrier_shape':[64,4],'fit':_jsonable(fit.metadata) if a.surface=='source' else None,'arrays':details,'surface':a.surface,'source_seal':seal,'frozen_fit':str(frozen),'frozen_fit_sha256':digest,'source_pack_sha256':source_pack_sha,'source_gate_sha256':source_gate_sha,'target_query_labels_used_for_carrier':False,'target_query_labels_used_for_selection':False}
 else:
  if a.profile_id=='muscle': raise ValueError('H1 has no muscle profile')
  cand={'old':'velocity7','encoding':'velocity7','state':'signed_state14'}[a.profile_id]
  dates=('1925-01-01','1925-01-08','1925-01-13','1925-01-15') if a.stage=='inner' else ('1925-01-01','1925-01-08','1925-01-13','1925-01-15','1925-01-19')
  target_dates=() if a.surface=='source' else (('1925-01-19',) if a.stage=='inner' else ('1925-01-20',))
  from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
  sources=tuple(s for d in dates for s in H1_SESSIONS_BY_DATE[d]);targets=tuple(s for d in target_dates for s in H1_SESSIONS_BY_DATE[d])
  data=ROOT.parent/'SPINT-main/data/000954'
  arrays={};details={}; source_pack_sha=None; source_gate_sha=None
  if a.surface=='source':
   rec=h1_profiles._load_records(data,sources); plan,diag=h1_profiles.fit_source_plan(rec,sources,cand)
  else:
   if a.source_pack is None or not a.source_pack.is_file() or a.source_gate is None or not a.source_gate.is_file(): raise RuntimeError('H1 target pack requires immutable source pack and explicit source gate')
   gate=json.loads(a.source_gate.read_text()); source_gate_sha=sha(a.source_gate)
   if gate.get('schema')!='carrier_last1_v2_paired_audit' or gate.get('status')!='PASSED' or gate.get('stage')!=a.stage or gate.get('dataset')!='h1': raise RuntimeError('H1 source gate identity drift')
   with np.load(a.source_pack,allow_pickle=False) as z:
    smeta=json.loads(str(z['metadata'].item())); source_pack_sha=sha(a.source_pack)
    if smeta.get('surface')!='source' or smeta.get('dataset')!='h1' or smeta.get('stage')!=a.stage or smeta.get('profile_id')!=a.profile_id: raise RuntimeError('source pack identity drift')
    plan=h1_profiles.ProfilePlan(cand,sources,tuple(smeta['plan']['source_input_sha256']),z['plan/behavior_rms'],z['plan/raw_basis'],z['plan/profile_mean'],z['plan/profile_scale'],smeta['plan']['source_raw_sha256'],smeta['plan']['source_profile_sha256'])
   rec=h1_profiles._load_records(data,targets)
  for s in (*(sources if a.surface=='source' else ()),*targets):
   z,_,d=h1_profiles.deploy_profile(rec[s],plan);arrays[f'carrier/{s}']=z;details[s]={**d,'sha256':ah(z)}
  if a.surface=='source':
   arrays.update({'plan/behavior_rms':plan.behavior_rms,'plan/raw_basis':plan.raw_basis,'plan/profile_mean':plan.profile_mean,'plan/profile_scale':plan.profile_scale})
  meta={'schema':'carrier_last1_v2_pack','dataset':'h1','profile_id':a.profile_id,'candidate':cand,'stage':a.stage,'sources':list(sources),'targets':list(targets),'carrier_shape':['variable',4],'plan':h1_profiles.plan_metadata(plan),'arrays':details,'surface':a.surface,'source_pack_sha256':source_pack_sha,'source_gate_sha256':source_gate_sha,'target_query_labels_used_for_carrier':False,'target_query_labels_used_for_selection':False,'target_record_materialized_for_support_projection':a.surface=='target'}
 arrays['metadata']=np.array(json.dumps(_jsonable(meta),sort_keys=True));atomic_npz(a.dest.resolve(),**arrays);a.dest.with_suffix('.json').write_text(json.dumps(_jsonable({**meta,'npz_sha256':sha(a.dest.resolve())}),indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
