#!/usr/bin/env python3
"""Receipt-bound seed-42 M2 mechanism-control summary; never trains or scores."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]; SELF=Path(__file__).resolve()
S={'ses-2020-10-30-Run1':519,'ses-2020-10-30-Run2':490,'ses-2020-11-18-Run1':425,'ses-2020-11-19-Run1':635}
SOURCE=('ses-2020-10-19-Run1','ses-2020-10-19-Run2','ses-2020-10-20-Run1','ses-2020-10-20-Run2','ses-2020-10-27-Run1','ses-2020-10-27-Run2','ses-2020-10-28-Run1')
JOINT_CELL='M2-RIFT-R50-D4-JOINT-FILM-M33-V1'; MECH_CELL='M2-RIFT-R50-MECHANISM-V1'; CACHE_FILES={'calib_activity.npy','X_store.npy','target_store.npy','T.npy','eligible_starts.npy','e0_u.pt'}
ARMS={'B':('joint','B_ACTIVITY_ONLY'),'C':('mechanism','C_CARRIER_ONLY'),'D':('joint','D_JOINT'),'shuffle':('mechanism','D_SHUFFLE'),'mean':('mechanism','D_JOINT'),'nonattn':('mechanism','D_JOINT')}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def finite(x):
 x=float(x)
 if not math.isfinite(x):raise ValueError
 return x
def run(root,k): return root/f"results/rift_v1/m2_r50_{'joint_'+k.lower() if k in ('B','D') else 'mechanism_'+k.lower()}_s42_formal_v1"
def tensor_map_ok(x,torch): return isinstance(x,dict) and bool(x) and all(torch.is_tensor(v) and v.numel() and (not(v.is_floating_point() or v.is_complex()) or bool(torch.isfinite(v).all())) for v in x.values())
def checkpoint_ok(p,meta,kind,epoch):
 try:
  import torch; st=torch.load(p,map_location='cpu',weights_only=False)
 except Exception as e:return f'CPU checkpoint load failed {p}: {e}'
 schema='m2_rift_joint_checkpoint_v2' if kind=='joint' else 'm2_rift_mechanism_checkpoint_v1'
 fixed={'schema':schema,'cell':meta['cell'],'arm':meta['arm'],'seed':42,'epoch':epoch,'smoke':False,'epochs':24,'context_bins':50,'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes']}
 if kind=='joint':fixed.update({'depth':4,'attention_backend':'local'})
 else:fixed.update({'aggregation':meta['aggregation'],'temporal':meta['temporal'],'raw_receptive_field':50})
 if not isinstance(st,dict) or any(st.get(k)!=v for k,v in fixed.items()) or st.get('global_step')!=epoch*3165:return f'checkpoint identity mismatch {p}'
 ema=st.get('ema',{})
 if not tensor_map_ok(st.get('model'),torch) or not isinstance(ema,dict) or ema.get('decay')!=.9995 or ema.get('n_updates')!=epoch*3165 or not tensor_map_ok(ema.get('shadow'),torch) or not set(ema['shadow']).issubset(st['model']):return f'checkpoint model/EMA invalid {p}'
 return None
def load(root,k):
 kind,arm=ARMS[k]; d=run(root,k); files={n:d/n for n in ('run_meta.json','train_receipt.json','score_receipt.json')}
 if any(not p.is_file() for p in files.values()):return None,[str(p) for p in files.values() if not p.is_file()]
 try:meta,train,score=(json.loads(p.read_text()) for p in files.values())
 except Exception as e:return None,[str(e)]
 ms='m2_rift_joint_train_v2' if kind=='joint' else 'm2_rift_mechanism_train_v1'; ts='m2_rift_joint_train_receipt_v2' if kind=='joint' else 'm2_rift_mechanism_train_receipt_v1'; ss='m2_rift_joint_ext4_epoch_scan_v1' if kind=='joint' else 'm2_rift_mechanism_ext4_epoch_scan_v1'
 cell=JOINT_CELL if kind=='joint' else MECH_CELL; train_bind=('cell','arm','seed','sampler_seed','epochs','source_hashes','cache_hashes'); score_bind=('cell','arm','seed','sampler_seed','source_hashes','cache_hashes')
 cache=meta.get('cache_hashes',{}); roster_ok=set(cache)=={'source_train','source_minival','ext4'} and set(cache.get('source_train',{}))==set(SOURCE) and set(cache.get('source_minival',{}))==set(SOURCE) and set(cache.get('ext4',{}))==set(S) and all(set(files)==CACHE_FILES for sessions in cache.values() for files in sessions.values())
 manifest_path=next((path for path in meta.get('source_hashes',{}) if path.endswith('shuffled_batch_manifest_24.json')),None)
 manifest_value=json.loads(Path(manifest_path).read_text()).get('digest') if manifest_path is not None and Path(manifest_path).is_file() else None
 digest=lambda x:isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
 if not(meta.get('schema')==ms and meta.get('status')=='FORMAL' and meta.get('cell')==cell and meta.get('arm')==arm and meta.get('seed')==42 and meta.get('sampler_seed')==42 and meta.get('epochs')==24 and meta.get('context_bins')==50 and meta.get('source_train_only_for_gradients') is True and meta.get('official_test_used') is False and isinstance(meta.get('source_hashes'),dict) and bool(meta['source_hashes']) and roster_ok and digest(meta.get('manifest_digest')) and meta['manifest_digest']==manifest_value and meta.get('optimizer',{}).get('total_updates')==75960 and meta.get('optimizer',{}).get('warmup_updates')==3165 and train.get('schema')==ts and train.get('status')=='COMPLETED' and train.get('global_step')==75960 and all(train.get(x)==meta.get(x) for x in train_bind) and score.get('schema')==ss and score.get('status')=='COMPLETED' and score.get('official_test_used') is False and all(score.get(x)==meta.get(x) for x in score_bind)):return None,[f'invalid receipt contract {d}']
 if kind=='mechanism' and (train.get('aggregation')!=meta.get('aggregation') or train.get('temporal')!=meta.get('temporal') or score.get('aggregation')!=meta.get('aggregation') or score.get('temporal')!=meta.get('temporal')):return None,[f'mechanism receipt aggregation/temporal drift {d}']
 if kind=='mechanism' and (meta.get('aggregation') != ('mean' if k=='mean' else 'slots') or meta.get('temporal') != ('nonattention' if k=='nonattn' else 'attention')):return None,[f'invalid mechanism control semantics {d}']
 rows=score.get('ema_by_epoch',{});
 if set(rows)!={str(x) for x in range(1,25)}:return None,[f'incomplete 24-row score {d}']
 for source,digest in meta['source_hashes'].items():
  if not Path(source).is_file() or sha(Path(source))!=digest:return None,[f'live source drift {source}']
 cache=root.parent/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'
 for surface,sessions in meta['cache_hashes'].items():
  for session,fs in sessions.items():
   for name,digest in fs.items():
    p=cache/surface/session/name
    if not p.is_file() or sha(p)!=digest:return None,[f'live cache drift {p}']
 import numpy as np
 for session in S:
  if tuple(np.load(cache/'ext4'/session/'calib_activity.npy',mmap_mode='r').shape)!=(33,100,96):return None,[f'M33 calibration shape drift {session}']
 means={}
 for e in range(1,25):
  row=rows[str(e)];p=d/f'epoch_{e:03d}.pt'
  if not p.is_file() or row.get('checkpoint_sha256')!=sha(p):return None,[f'checkpoint hash drift {p}']
  err=checkpoint_ok(p,meta,kind,e)
  if err:return None,[err]
  try:
   values=[finite(row['per_session'][s]['r2']) for s in S]; mean=finite(row['equal_session_mean'])
   if row.get('partial') is not False or row.get('n_windows')!=2069 or set(row.get('per_session',{}))!=set(S) or any(row['per_session'][s].get('window_count')!=n for s,n in S.items()) or abs(mean-sum(values)/4)>1e-12:raise ValueError
  except (KeyError,TypeError,ValueError):return None,[f'invalid ext4 row {p}']
  means[e]=mean
 best=min(range(1,25),key=lambda e:(-means[e],e))
 if score.get('selection')!={'rule':'earliest best equal_session_mean','epoch':best,'equal_session_mean':means[best]} or score.get('endpoint24')!=rows['24']:return None,[f'bad selection/endpoint {d}']
 return {'meta':meta,'score':score,'best':best,'run':str(d),'receipt_sha256':{n:sha(p) for n,p in files.items()},'checkpoint_sha256_by_epoch':{str(e):sha(d/f'epoch_{e:03d}.pt') for e in range(1,25)}},[]
def summary(root):
 base={'schema':'m2_mechanism_controls_summary_v1','summarizer_sha256':sha(SELF),'seed':42,'surface':'fixed ext4 M33, four sessions / 2069 windows, visible development only','official_test_used':False,'no_ci_pvalue_or_multiseed_architecture_effect':True,'calibration_information':{'B':'activity-only calibration information','C':'carrier-only calibration information; query activity remains used','D':'joint activity-plus-carrier calibration information'}}
 absent=[str(run(root,k)/name) for k in ARMS for name in ('run_meta.json','train_receipt.json','score_receipt.json') if not (run(root,k)/name).is_file()]
 if absent:return {**base,'status':'PENDING','missing_or_invalid':sorted(absent),'completed_arms':[],'reason':'all B/C/D/shuffle/mean/nonattn formal 24-epoch receipt triplets are required before effects'}
 rows={};errors=[]
 for k in ARMS:
  x,e=load(root,k);errors+=e
  if x:rows[k]=x
 if errors:return {**base,'status':'PENDING','missing_or_invalid':sorted(errors),'completed_arms':sorted(rows),'reason':'all B/C/D/shuffle/mean/nonattn formal 24-epoch receipt triplets are required before effects'}
 common=rows['B']['meta']['cache_hashes']
 if any(rows[k]['meta']['cache_hashes']!=common for k in rows):return {**base,'status':'PENDING','missing_or_invalid':['shared cache input drift'],'completed_arms':sorted(rows)}
 shared=set.intersection(*(set(row['meta']['source_hashes']) for row in rows.values()))
 if not shared or any(len({row['meta']['source_hashes'][path] for row in rows.values()}) != 1 for path in shared):return {**base,'status':'PENDING','missing_or_invalid':['shared runtime source input drift'],'completed_arms':sorted(rows)}
 if len({row['meta'].get('manifest_digest') for row in rows.values()})!=1:return {**base,'status':'PENDING','missing_or_invalid':['frozen manifest digest drift'],'completed_arms':sorted(rows)}
 def delta(left,right,el,er):
  a,b=rows[left]['score']['ema_by_epoch'][str(el)],rows[right]['score']['ema_by_epoch'][str(er)]
  return {'D_minus_control_equal_session_mean':finite(a['equal_session_mean'])-finite(b['equal_session_mean']),'per_session_D_minus_control':{s:finite(a['per_session'][s]['r2'])-finite(b['per_session'][s]['r2']) for s in S}}
 effects={k:{'independently_picked':delta('D',k,rows['D']['best'],rows[k]['best']),'fixed_epoch_24':delta('D',k,24,24)} for k in ('shuffle','mean','nonattn')}
 def arm_row(v):
  score=v['score']; pick=score['ema_by_epoch'][str(v['best'])]
  return {'run':v['run'],'selected_epoch':v['best'],'selected':pick,'endpoint24':score['endpoint24'],'receipt_sha256':v['receipt_sha256'],'checkpoint_sha256_by_epoch':v['checkpoint_sha256_by_epoch'],'source_hashes':v['meta']['source_hashes'],'cache_hashes':v['meta']['cache_hashes']}
 return {**base,'status':'COMPLETED','frozen_manifest_digest':rows['B']['meta']['manifest_digest'],'arms':{k:arm_row(v) for k,v in rows.items()},'information_contrasts':{'D_minus_B_independently_picked':delta('D','B',rows['D']['best'],rows['B']['best']),'D_minus_C_independently_picked':delta('D','C',rows['D']['best'],rows['C']['best']),'D_minus_B_fixed_epoch_24':delta('D','B',24,24),'D_minus_C_fixed_epoch_24':delta('D','C',24,24)},'effects':effects,'interpretation':'Seed42 development evidence only; sessions are within-seed repeated measurements.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();
 if a.output.exists():raise FileExistsError(f'refusing to overwrite existing output {a.output}')
 x=summary(a.root.resolve());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':x['status'],'output':str(a.output)}))
if __name__=='__main__':main()
