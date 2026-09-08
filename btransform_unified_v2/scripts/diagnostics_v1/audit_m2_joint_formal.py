#!/usr/bin/env python3
"""Read-only formal M2 joint receipt/checkpoint audit."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
CELL='M2-RIFT-R50-D4-JOINT-FILM-M33-V1'; WINDOWS={'ses-2020-10-30-Run1':519,'ses-2020-10-30-Run2':490,'ses-2020-11-18-Run1':425,'ses-2020-11-19-Run1':635}
SOURCE=('ses-2020-10-19-Run1','ses-2020-10-19-Run2','ses-2020-10-20-Run1','ses-2020-10-20-Run2','ses-2020-10-27-Run1','ses-2020-10-27-Run2','ses-2020-10-28-Run1'); CACHE_FILES={'calib_activity.npy','X_store.npy','target_store.npy','T.npy','eligible_starts.npy','e0_u.pt'}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text())
 if not isinstance(x,dict):raise RuntimeError(f'object required: {p}')
 return x
def finite(x:Any)->float:
 x=float(x)
 if not math.isfinite(x):raise ValueError
 return x
def tensors(x:Any,torch:Any)->bool:
 return isinstance(x,dict) and bool(x) and all(torch.is_tensor(v) and v.numel()>0 and (not(v.is_floating_point() or v.is_complex()) or bool(torch.isfinite(v).all())) for v in x.values())
def cp_check(p:Path,meta:dict[str,Any],arm:str,seed:int,e:int)->str|None:
 try:
  import torch; x=torch.load(p,map_location='cpu',weights_only=False)
 except Exception as exc:return f'CPU load {p}: {exc}'
 expect={'schema':'m2_rift_joint_checkpoint_v2','cell':CELL,'arm':arm,'seed':seed,'epoch':e,'global_step':e*3165,'smoke':False,'epochs':24,'context_bins':50,'depth':4,'attention_backend':'local','source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes']}
 if not isinstance(x,dict) or any(x.get(k)!=v for k,v in expect.items()):return f'identity mismatch {p}'
 ema=x.get('ema',{})
 if not tensors(x.get('model'),torch) or not isinstance(ema,dict) or ema.get('decay')!=.9995 or ema.get('n_updates')!=e*3165 or not tensors(ema.get('shadow'),torch) or not set(ema['shadow']).issubset(x['model']):return f'nonfinite/empty raw EMA {p}'
 return None
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm',choices=('B','D'),required=True);p.add_argument('--seed',choices=(42,43,44),type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
 if out.exists():raise FileExistsError(f'refusing existing output {out}')
 arm='B_ACTIVITY_ONLY' if a.arm=='B' else 'D_JOINT';run=root/f'results/rift_v1/m2_r50_joint_{a.arm.lower()}_s{a.seed}_formal_v1';paths={n:run/n for n in ('run_meta.json','train_receipt.json','score_receipt.json','ext4_epoch_scan.json')}
 try:
  meta,train,score,scan=(load(paths[n]) for n in paths)
  checks=[]
  def require(label:str,ok:bool):
   checks.append({'name':label,'passed':bool(ok)})
   if not ok:raise RuntimeError(label)
  cache_meta=meta.get('cache_hashes',{}); roster=set(cache_meta)=={'source_train','source_minival','ext4'} and set(cache_meta.get('source_train',{}))==set(SOURCE) and set(cache_meta.get('source_minival',{}))==set(SOURCE) and set(cache_meta.get('ext4',{}))==set(WINDOWS) and all(set(files)==CACHE_FILES for sessions in cache_meta.values() for files in sessions.values())
  mp=next((Path(x) for x in meta.get('source_hashes',{}) if x.endswith('shuffled_batch_manifest_24.json')),None); md=load(mp).get('digest') if mp and mp.is_file() else None
  require('meta',all(meta.get(k)==v for k,v in {'schema':'m2_rift_joint_train_v2','status':'FORMAL','cell':CELL,'arm':arm,'seed':a.seed,'sampler_seed':42,'epochs':24,'context_bins':50,'depth':4,'attention_backend':'local','official_test_used':False,'source_train_only_for_gradients':True}.items()) and meta.get('optimizer',{}).get('total_updates')==75960 and meta.get('optimizer',{}).get('warmup_updates')==3165 and isinstance(meta.get('source_hashes'),dict) and bool(meta['source_hashes']) and roster and isinstance(meta.get('manifest_digest'),str) and len(meta['manifest_digest'])==64 and meta['manifest_digest']==md)
  require('train',all(train.get(k)==v for k,v in {'schema':'m2_rift_joint_train_receipt_v2','status':'COMPLETED','cell':CELL,'arm':arm,'seed':a.seed,'sampler_seed':42,'epochs':24,'global_step':75960,'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes']}.items()))
  require('score_scan_equal',score==scan)
  require('score',all(score.get(k)==v for k,v in {'schema':'m2_rift_joint_ext4_epoch_scan_v1','status':'COMPLETED','cell':CELL,'arm':arm,'seed':a.seed,'sampler_seed':42,'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes'],'official_test_used':False}.items()))
  for source,digest in meta['source_hashes'].items():require(f'source:{source}',Path(source).is_file() and sha(Path(source))==digest)
  cache=root.parent/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'
  for surface,sessions in meta['cache_hashes'].items():
   for session,files in sessions.items():
    for name,digest in files.items():
     q=cache/surface/session/name;require(f'cache:{surface}/{session}/{name}',q.is_file() and sha(q)==digest)
  rows=score.get('ema_by_epoch',{});require('all24rows',set(rows)=={str(e) for e in range(1,25)})
  cps={}
  for e in range(1,25):
   q=run/f'epoch_{e:03d}.pt';h=sha(q);cps[str(e)]=h;row=rows[str(e)];require(f'cpSHA:{e}',q.is_file() and row.get('checkpoint_sha256')==h);err=cp_check(q,meta,arm,a.seed,e);require(f'cpIdentity:{e}: {err or "passed"}',err is None);per=row.get('per_session',{});vals=[finite(per[s]['r2']) for s in WINDOWS];require(f'ext4:{e}',row.get('partial') is False and row.get('n_windows')==2069 and set(per)==set(WINDOWS) and all(per[s].get('window_count')==n for s,n in WINDOWS.items()) and abs(finite(row['equal_session_mean'])-sum(vals)/4)<=1e-12)
  best=min(range(1,25),key=lambda e:(-finite(rows[str(e)]['equal_session_mean']),e));require('selection',score.get('selection')=={'rule':'earliest best equal_session_mean','epoch':best,'equal_session_mean':finite(rows[str(best)]['equal_session_mean'])});require('endpoint24',score.get('endpoint24')==rows['24'])
  state=root/f'results/paper_program_v1/m2_r50_joint_{a.arm.lower()}_s{a.seed}_formal_v1/state.json';st=load(state);spec=Path(st.get('spec',''));sp=load(spec) if spec.is_file() else {};require('guard',st.get('status')=='COMPLETE' and st.get('cell')==f'm2_r50_joint_{a.arm.lower()}_s{a.seed}_formal_v1' and str(run) in st.get('argv',[]) and st.get('spec_sha256')==sha(spec) and st.get('cell')==sp.get('cell') and str(run) in sp.get('new_destinations',[]))
  report={'schema':f'm2_joint_{a.arm.lower()}_s{a.seed}_formal_validation_v1','status':'PASSED','failures':[],'run':str(run),'inputs':{n:sha(q) for n,q in paths.items()},'helper_source_sha256':sha(Path(__file__)),'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes'],'guard_state':{'path':str(state),'sha256':sha(state),'status':st['status'],'spec_path':str(spec),'spec_sha256':sha(spec)},'checks':checks,'checkpoint_sha256_by_epoch':cps,'selected_ext4':{'epoch':best,**rows[str(best)]},'epoch_24_ext4':rows['24']}
 except Exception as exc:
  report={'schema':f'm2_joint_{a.arm.lower()}_s{a.seed}_formal_validation_v1','status':'FAILED','failures':[str(exc)]}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'output':str(out)}))
if __name__=='__main__':main()
