#!/usr/bin/env python3
"""Build H1 C2 banks with the source-frozen carrier-v4 common estimator."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np,torch
from common import DATA,SOURCE_PAYLOAD,atomic_json,index_heldout_calib,jsonable,load_public_heldout_m3,setup_imports,sha_array,sha_file
HERE=Path(__file__).resolve().parent;V4=HERE.parent;sys.path.insert(0,str(V4));import common_estimator as core
setup_imports()
from btransform_unified_v1 import adapters,h1_config
from btransform_unified_v1.c2_protocol import HELDOUT_SESSION_TO_FALCON_KEY
import h1_profiles

def records_for_fit(records):
 out={}
 for s,r in records.items():
  # Full source behavior excludes the final two source-validation trials;
  # support remains the legal first M3 velocity/rate rows.
  vals=tuple(float(x) for x in r.trial_values);fitvals=vals[:-2]
  fit=np.concatenate([h1_profiles.support_blocks(r,(v,))[1] for v in fitvals],axis=0)
  sup=h1_profiles.support_blocks(r,vals[:3])[1]
  rates=h1_profiles.support_blocks(r,vals[:3])[0]
  out[s]={"fit_behavior":fit,"support_behavior":sup,"support_rates":rates}
 return out
def deploy_record(fit,record,trials):
 r,b,_used=h1_profiles.support_blocks(record,trials)
 return core.deploy(fit,b,r)
def save_fit(fit,path_prefix):return core.save_fit(fit,path_prefix)
def load_fit(json_path):return core.load_fit(json_path)
ARMS=("full","activity_only","carrier_only","none")
def materialize(activity,carrier,information_arm="full",materializer=None):
 if materializer is None:
  from h1_c2_cal1_b2_l200_p16 import _materialize_e0
  materializer=_materialize_e0
 zero=np.zeros_like(carrier,dtype=np.float32)
 if information_arm=="full": e0,hc=materializer(activity,carrier)
 elif information_arm=="activity_only": e0,hc=materializer(activity,zero)
 elif information_arm=="carrier_only":
  from btransform_unified_v1.h1_config import FULL_E0_DIM
  e0,hc=np.zeros((carrier.shape[0],FULL_E0_DIM),np.float32),carrier
 elif information_arm=="none":
  from btransform_unified_v1.h1_config import FULL_E0_DIM
  e0,hc=np.zeros((carrier.shape[0],FULL_E0_DIM),np.float32),zero
 else: raise ValueError(f"unknown information arm {information_arm}")
 if information_arm in ('full','carrier_only') and not np.array_equal(hc,carrier):raise RuntimeError('C2 carrier rematerialization drift')
 return e0,hc
def arm_invariant_smoke():
 """No-data check of the exact `(E0,T)` law used by both bank surfaces."""
 a=np.ones((3,2,4),np.float32);t=np.arange(8,dtype=np.float32).reshape(2,4)+1
 def fake(activity,carrier):return np.full((2,700),float(carrier.sum()),np.float32),np.asarray(carrier,np.float32)
 rows={arm:materialize(a,t,arm,fake) for arm in ARMS}
 assert np.array_equal(rows['full'][1],t) and np.array_equal(rows['carrier_only'][1],t)
 assert not np.array_equal(rows['full'][0],rows['activity_only'][0])
 for arm in ('activity_only','none'):assert not np.count_nonzero(rows[arm][1])
 for arm in ('carrier_only','none'):assert not np.count_nonzero(rows[arm][0])
 return {'status':'PASSED','arms':list(ARMS)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dest',type=Path);ap.add_argument('--information-arm',choices=ARMS,default='full');ap.add_argument('--fit-json',type=Path);ap.add_argument('--arm-invariant-smoke',action='store_true');a=ap.parse_args()
 if a.arm_invariant_smoke: print(json.dumps(arm_invariant_smoke()));return
 if a.dest is None: ap.error('--dest is required unless --arm-invariant-smoke')
 dest=a.dest.resolve()
 if dest.exists():raise FileExistsError(dest)
 sources=tuple(h1_config.H1_ALL_SESSIONS);need=len(sources)==13
 if not need:raise RuntimeError('official H1 source roster drift')
 legacy=h1_profiles._legacy();paths=legacy.index_heldin_calib(DATA);records=h1_profiles._load_records(DATA,sources)
 if a.fit_json:
  fit_json=a.fit_json.resolve();fit=core.load_fit(fit_json);fit_npz=Path(fit.metadata['npz_path']);diag={'reused_fit_json_sha256':sha_file(fit_json)}
  if fit.task!='h1' or tuple(fit.source_sessions)!=sources:raise RuntimeError('reused V4 fit task/source13 drift')
 else:
  fit,source_carriers,diag=core.fit_source('h1',records_for_fit(records));dest.mkdir(parents=True);fit_json,fit_npz=core.save_fit(fit,dest/'latent_state3_ridge_intercept')
 if not dest.exists():dest.mkdir(parents=True)
 payload=torch.load(SOURCE_PAYLOAD,map_location='cpu',weights_only=False);rows=payload.get('sessions',{});
 if len(rows)!=27:raise RuntimeError('official payload must have 27 tags')
 heldout=index_heldout_calib();fmap={s:k for s,k in HELDOUT_SESSION_TO_FALCON_KEY};arrays={};details={}
 for tag,row in sorted(rows.items()):
  session=str(row['session']);trials=tuple(float(x) for x in row['calibration_trials'])
  if len(trials)!=3 or len(set(trials))!=3:raise RuntimeError(f'{tag}: nonofficial M3')
  if session in paths:record=records[session];surface='held-in-calib';path=paths[session]
  else:record=load_public_heldout_m3(heldout[session],legacy);surface='held-out-calib';path=heldout[session]
  if tuple(record.trial_values[:3])!=trials:raise RuntimeError(f'{tag}: M3 trial drift')
  t,raw,info=deploy_record(fit,record,trials);activity,old=adapters._h1_payload_arrays(session);e0,hc=materialize(activity,t,a.information_arm)
  if e0.shape!=(176,700) or hc.shape!=(176,4):raise RuntimeError(f'{tag}: arm E0/T geometry drift {e0.shape}/{hc.shape}')
  arrays[f'T/{tag}']=np.ascontiguousarray(hc,np.float32);arrays[f'E0/{tag}']=e0;arrays[f'raw/{tag}']=raw
  effective_t=arrays[f'T/{tag}']
  if a.information_arm in ('activity_only','none') and np.count_nonzero(effective_t):raise RuntimeError(f'{tag}: carrier leaked into {a.information_arm}')
  if a.information_arm in ('carrier_only','none') and np.count_nonzero(e0):raise RuntimeError(f'{tag}: activity leaked into {a.information_arm}')
  details[tag]={"session":session,"surface":surface,"falcon_key":fmap.get(session),"trials":list(trials),"path":str(path),"path_sha256":sha_file(path),"T_sha256":sha_array(effective_t),"E0_sha256":sha_array(e0),"full_T_sha256":sha_array(t),"old_hc_T_sha256":sha_array(np.asarray(old,np.float32)),"activity_sha256":sha_array(activity),"direct_carrier_byte_equal":bool(np.array_equal(hc,effective_t)),"information_arm":a.information_arm,**jsonable(info)}
 np.savez_compressed(dest/'banks_27.npz',**arrays)
 receipt={"schema":"h1_carrier_v4_latent_state3_ridge_intercept_27tag_v1","status":"BUILT","utc":datetime.now(timezone.utc).isoformat(),"candidate":"latent_state3_ridge_intercept/per_column","estimator":core.SCHEMA,"information_arm":a.information_arm,"source_sessions":list(sources),"source_count":13,"tag_count":27,"support":"official first-three calibration trials; C2 M3 activity unchanged","fit_json":str(fit_json),"fit_json_sha256":sha_file(fit_json),"fit_npz":str(fit_npz),"fit_npz_sha256":sha_file(fit_npz),"banks_27":str((dest/'banks_27.npz').resolve()),"banks_27_sha256":sha_file(dest/'banks_27.npz'),"source_payload":str(SOURCE_PAYLOAD),"source_payload_sha256":sha_file(SOURCE_PAYLOAD),"official_test_used":False,"last_date_0564_used":False,"tags":details,"source_diagnostics":jsonable(diag),"implementation_sha256":{str(p):sha_file(p) for p in (Path(__file__).resolve(),HERE/'common.py',V4/'common_estimator.py')}}
 atomic_json(dest/'receipt.json',receipt);print(json.dumps({'receipt':str(dest/'receipt.json')}))
if __name__=='__main__':main()
