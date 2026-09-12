"""Raw-bin FAIR V2 data plane; support selection never reads target Y."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parent; WS=ROOT.parent
for p in (ROOT/'learnable_recency_v1/scripts', ROOT/'learnable_recency_v1/src', ROOT/'src', WS/'btransform_unified_v1/src', WS/'SPINT-main', WS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

def _sha_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def _sha(a:Any)->str:
 x=np.ascontiguousarray(np.asarray(a));h=hashlib.sha256();h.update(x.dtype.str.encode());h.update(str(x.shape).encode());h.update(x.tobytes());return h.hexdigest()
def _item(X,Y,starts,pad,support,segments,prov):
 X=np.ascontiguousarray(X,np.float32); starts=np.ascontiguousarray(starts,np.int64); Y=np.ascontiguousarray(Y,np.float32)
 raw=X[int(pad):]; ix=np.ascontiguousarray(support,np.int64)
 if raw.ndim!=2 or np.any(ix<0) or np.any(ix>=len(raw)) or not np.array_equal(raw[ix],np.asarray(prov.pop('_support_raw'))): raise RuntimeError('support/raw coordinate drift')
 return {'X':X,'Y':Y,'starts':starts,'pad':int(pad),'support_indices':ix,'support':np.ascontiguousarray(raw[ix],np.float32),'support_segments':[np.ascontiguousarray(x,np.int64) for x in segments],'support_provenance':prov}

def _m1_trial_support(item:dict, path:Path)->tuple[np.ndarray,list[np.ndarray],dict]:
 import h5py
 raw=np.asarray(item['X'],np.float32)[99:]
 with h5py.File(path,'r') as f:
  ts=np.asarray(f['acquisition/eval_mask/timestamps']); ev=np.asarray(f['acquisition/eval_mask/data'],bool); starts=np.asarray(f['intervals/trials/start_time'])
 changes=np.zeros(len(raw),bool);changes[np.searchsorted(ts,starts,side='left').clip(0,len(raw)-1)]=True
 # Existing activity route drops eval-invalid bins before trializing; map its first
 # ten trial starts back to original raw coordinates without interpolation.
 valid_ix=np.flatnonzero(ev); kept_changes=changes[ev]; all_trial_lo=valid_ix[np.flatnonzero(kept_changes)]; trial_lo=all_trial_lo[:10]
 if len(trial_lo)!=10: raise RuntimeError(f'{path}: fewer than ten eval-valid trials')
 seg=[]
 for i,lo in enumerate(trial_lo):
  hi=all_trial_lo[i+1] if i+1<len(all_trial_lo) else len(raw); z=np.flatnonzero(ev[lo:hi])+lo
  if not len(z): raise RuntimeError(f'{path}: empty eval-valid trial')
  seg.append(z.astype(np.int64))
 ix=np.concatenate(seg); return ix,seg,{'raw_nwb':str(path),'raw_nwb_sha256':_sha_file(path),'trial_ids':list(range(10)),'budget':'first10 eval-valid raw trials','native_bins':int(len(ix))}

def _m1(include:bool):
 import m1_static_data as d
 src=d.load_source()['items']; ho=d.load_heldout()['items'] if include else {}
 def conv(session,item,role):
  path=Path(item['file']); ix,segs,prov=_m1_trial_support(item,path)
  prov['_support_raw']=np.asarray(item['X'])[99:][ix]
  return _item(item['X'],item['Y'],item['starts'],99,ix,segs,prov)
 out={'train':{s:conv(s,v,'train') for s,v in src.items()},'evaluation':{s:conv(s,v,'eval') for s,v in ho.items()}}
 source_windows=sum(len(x['Y']) for x in out['train'].values()); target_windows=sum(len(x['Y']) for x in out['evaluation'].values())
 if source_windows!=213336 or (include and target_windows!=3881): raise RuntimeError('M1 static window-count contract drift')
 out['metadata']={'task':'m1','context':100,'units':64,'outputs':16,'source_windows':source_windows,'target_windows':target_windows if include else 3881}
 return out

def _m2_trial_support(session:str, item:dict, role:str)->tuple[np.ndarray,list[np.ndarray],dict]:
 """Return mask-valid native bins from trial IDs 0..32 on the query timeline.

 The historical M33 tensor had 100 cubic-resampled bins per trial.  This
 route deliberately uses neither that tensor nor its offsets: cache ``X`` is
 the canonical padded query timeline, and NWB timestamps identify its first
 33 original trial intervals.
 """
 import h5py
 root=WS/'SPINT-main/data/000953'
 token='held-in-calib' if role=='source' else 'held-out-calib'
 found=sorted(root.rglob(f'*{token}*{session}*.nwb'))
 if len(found)!=1: raise RuntimeError(f'{session}: expected one {token} NWB, got {found}')
 path=found[0]; raw=np.asarray(item['X'],np.float32)[49:]
 from falcon_challenge.config import FalconTask
 from falcon_challenge.dataloaders import load_nwb
 neural, _covariates, _trial_change, loader_eval=load_nwb(path,FalconTask.m2)
 neural=np.asarray(neural,np.float32)
 with h5py.File(path,'r') as f:
  ts=np.asarray(f['acquisition/finger_vel/mrs/timestamps'],np.float64)
  ev=np.asarray(f['acquisition/eval_mask/data'],bool)
  trial_ids=np.asarray(f['intervals/trials/id'],np.int64)
  starts=np.asarray(f['intervals/trials/start_time'],np.float64); stops=np.asarray(f['intervals/trials/stop_time'],np.float64)
 if len(ts)!=len(raw) or ev.shape!=(len(raw),) or not np.array_equal(trial_ids,np.arange(len(trial_ids),dtype=np.int64)):
  raise RuntimeError(f'{session}: raw-NWB/query-cache timeline contract drift')
 if not np.array_equal(neural,raw) or not np.array_equal(np.asarray(loader_eval,bool),ev):
  raise RuntimeError(f'{session}: raw-NWB neural/query-cache byte contract drift')
 if len(starts)<33 or np.any(np.diff(ts)<=0) or np.any(stops[:33]<=starts[:33]): raise RuntimeError(f'{session}: invalid M33 trial metadata')
 seg=[]
 for lo_t,hi_t in zip(starts[:33],stops[:33]):
  lo=int(np.searchsorted(ts,lo_t,'left')); hi=int(np.searchsorted(ts,hi_t,'left'))
  z=np.flatnonzero(ev[lo:hi])+lo
  if not len(z): raise RuntimeError(f'{session}: empty eval-valid support trial')
  seg.append(z.astype(np.int64))
 ix=np.concatenate(seg)
 return ix,seg,{'raw_nwb':str(path),'raw_nwb_sha256':_sha_file(path),'trial_ids':list(range(33)),'budget':'first33 eval-mask-valid native raw trials','native_bins':int(len(ix)),'query_raw_bins':int(len(raw)),'raw_nwb_neural_equals_query_raw':True}
def _m2(include:bool):
 import m2_static_train as tr, m2_static_score as score
 src=tr.load_static_surface('source_train')
 from scripts.rift_v1 import m2_ext6_epoch_pick as frozen
 ev=score.query_surface(frozen.QUERY_CACHE) if include else {}
 def source(s,v):
  ix,segs,p=_m2_trial_support(s,v,'source'); p['_support_raw']=np.asarray(v['X'])[49:][ix];return _item(v['X'],v['Y'],v['starts'],49,ix,segs,p)
 out={'train':{s:source(s,v) for s,v in src.items()},'evaluation':{}}
 for s,v in ev.items():
  ix,segs,p=_m2_trial_support(s,v,'evaluation'); p['_support_raw']=np.asarray(v['X'])[49:][ix]
  out['evaluation'][s]=_item(v['X'],v['Y'],v['starts'],49,ix,segs,p)
 source_windows=sum(len(x['Y']) for x in out['train'].values()); target_windows=sum(len(x['Y']) for x in out['evaluation'].values())
 if source_windows!=101171 or (include and target_windows!=15403): raise RuntimeError('M2 static window-count contract drift')
 out['metadata']={'task':'m2','context':50,'units':96,'outputs':2,'source_windows':source_windows,'target_windows':target_windows if include else 15403}
 return out

def _h1(include:bool):
 from btransform_unified_v1 import adapters,h1_config
 from falcon_challenge.config import FalconTask
 from falcon_challenge.dataloaders import load_nwb
 from pynwb import NWBHDF5IO
 def make(session, neural, y, starts, path, ev, tid):
  raw=np.asarray(neural,np.float32); X=np.pad(raw,((299,0),(0,0))); ix=[]; ids=[]
  for z in tid[ev & np.isfinite(tid)]:
   if not ids or z!=ids[-1]:ids.append(float(z))
  ids=ids[:3]
  if len(ids)!=3: raise RuntimeError(f'{path}: fewer than three eval-valid TrialNum runs')
  seg=[np.flatnonzero(ev&(tid==z)).astype(np.int64) for z in ids]; support=np.concatenate(seg)
  if any(not len(x) for x in seg): raise RuntimeError(f'{path}: empty eval-valid TrialNum run')
  p={'raw_nwb':str(path),'raw_nwb_sha256':_sha_file(Path(path)),'trial_ids':ids,'budget':'first3 eval-valid TrialNum raw trials','native_bins':int(len(support)),'_support_raw':raw[support]}
  return _item(X,np.asarray(y,np.float32)[starts],starts,299,support,seg,p)
 cache=adapters._h1_source_cache()['train']; train={}
 root=WS/'SPINT-main/data/000954/sub-HumanPitt-held-in-calib'
 for s in h1_config.H1_ALL_SESSIONS:
  row=cache[s]; path=next(root.glob(f'*_ses-{s.removeprefix("ses-")}*.nwb'))
  import h5py
  with h5py.File(path,'r') as f: tid=np.asarray(f['acquisition/TrialNum/data']);ev=np.asarray(f['acquisition/eval_mask/data'],bool)
  ends=np.asarray(row['query_starts'])+h1_config.FULL_WINDOW-1;ends=ends[np.asarray(row['eval_mask'],bool)[ends]];train[s]=make(s,row['neural'],row['velocity'],ends,path,ev,tid)
 out={'train':train,'evaluation':{}}
 if include:
  root=WS/'SPINT-main/data/000954/sub-HumanPitt-held-out-calib'
  for path in sorted(root.glob('*.nwb')):
   neural,vel,_c,ev=load_nwb(path,FalconTask.h1);ev=np.asarray(ev,bool)
   with NWBHDF5IO(str(path),'r',load_namespaces=True) as f:tid=np.asarray(f.read().acquisition['TrialNum'].data[:])
   ends=np.flatnonzero(ev).astype(np.int64);s=path.stem.split('_ses-',1)[1];out['evaluation'][s]=make(s,neural,vel,ends,path,ev,tid)
 source_windows=sum(len(x['Y']) for x in out['train'].values()); target_windows=sum(len(x['Y']) for x in out['evaluation'].values())
 if source_windows!=23212 or (include and target_windows!=33613): raise RuntimeError(f'H1 static window-count contract drift: source={source_windows}, target={target_windows}')
 out['metadata']={'task':'h1','context':300,'units':176,'outputs':7,'source_windows':source_windows,'target_windows':target_windows if include else 33613}
 return out

def load_task(task:str,*,include_evaluation:bool=False)->dict[str,Any]:
 task=task.lower()
 if task=='m1':return _m1(include_evaluation)
 if task=='m2':return _m2(include_evaluation)
 if task=='h1':return _h1(include_evaluation)
 raise ValueError(task)
