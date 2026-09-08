"""Query-only M2 cache access and chronological source-trial split."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
from btransform_unified_v1.bank import TaskBank, array_sha256
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT.parent/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'
DATA=ROOT.parent/'SPINT-main/data/000953'
EPOCHS=24; BATCH=32; SPLIT_FRACTION=.20; SPLIT_SEED=20260908
@dataclass(frozen=True)
class QueryBank:
 session_id:str; X_store:np.ndarray; target_store:np.ndarray; eligible_starts:np.ndarray; unit_mask:torch.Tensor
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def zero_task_bank(s:str)->TaskBank:
 e=np.zeros((96,50),np.float32);t=np.zeros((96,4),np.float32)
 return TaskBank(s,e,t,np.ones(96,np.bool_),np.empty((0,50,96),np.float32),np.empty((0,2),np.float32),np.empty(0,np.int64),{'shape':[96,50],'trial_count':33,'estimator':'cross_session_query_only_zero_bank','array_sha256':array_sha256(e),'budget':33,'cache_root':str(CACHE)})
def query_surface(surface:str,device:torch.device):
 sessions=old_plan.EXT4_SESSIONS if surface=='ext4' else old_plan.HELDIN_SESSIONS; q={};t={}
 for s in sessions:
  root=CACHE/surface/s;x=np.load(root/'X_store.npy',mmap_mode='r');y=np.load(root/'target_store.npy',mmap_mode='r');starts=np.load(root/'eligible_starts.npy',mmap_mode='r')
  if x.ndim!=2 or x.shape[1]!=96 or y.shape!=(len(starts),2):raise RuntimeError(f'{surface}/{s}: query geometry')
  q[s]=QueryBank(s,x,y,starts,torch.ones(96,dtype=torch.bool,device=device));t[s]=zero_task_bank(s)
 return q,t
def _source_nwb(s:str)->Path:
 found=sorted(DATA.rglob(f'*held-in-calib*{s}*.nwb'))
 if len(found)!=1:raise RuntimeError(f'{s}: expected one held-in-calib NWB, got {found}')
 return found[0]
def _trial_bins(s:str,raw_bins:int):
 p=_source_nwb(s)
 with h5py.File(p,'r') as f:
  ids=np.asarray(f['intervals/trials/id'],np.int64);starts=np.asarray(f['intervals/trials/start_time'],np.float64);stops=np.asarray(f['intervals/trials/stop_time'],np.float64);times=np.asarray(f['acquisition/finger_vel/mrs/timestamps'],np.float64)
 if len(times)!=raw_bins or not np.array_equal(ids,np.arange(len(ids),dtype=np.int64)):raise RuntimeError(f'{s}: trial/timestamp source contract drift')
 # The canonical loader uses timestamp half-open intervals.  Trial events need
 # not fall exactly on a discrete covariate sample, so require monotonicity,
 # range containment, and at most one local time-bin of insertion error rather
 # than an invalid microsecond-equality assertion.
 if not np.all(np.diff(times)>0) or np.any(stops<=starts):raise RuntimeError(f'{s}: nonmonotone timestamp/trial contract')
 dt=float(np.median(np.diff(times)));range_tol=1e-8;first=np.searchsorted(times,starts,'left');last=np.searchsorted(times,stops,'left')
 if np.any(first>=raw_bins) or np.any(last<=first) or np.any(starts<times[0]-dt-range_tol) or np.any(stops>times[-1]+dt+range_tol):raise RuntimeError(f'{s}: invalid trial interval range')
 if np.any(times[first]-starts>dt+1e-9):raise RuntimeError(f'{s}: trial-start insertion exceeds one bin')
 interior=last<raw_bins
 if np.any(times[last[interior]]-stops[interior]>dt+1e-9):raise RuntimeError(f'{s}: trial-stop insertion exceeds one bin')
 return first.astype(np.int64),last.astype(np.int64),p
def source_train_val_split(device:torch.device):
 full,tasks=query_surface('source_train',device);train={};val={};rows={}
 for s,b in full.items():
  root=CACHE/'source_train'/s;mapping=json.loads((root/'mapping.json').read_text())
  if tuple(mapping.get('support_trial_ids',()))!=tuple(range(33)) or int(mapping.get('query_pad_bins',-1))!=49:raise RuntimeError(f'{s}: M33/query padding drift')
  trial_start,trial_stop,nwb=_trial_bins(s,int(b.X_store.shape[0])-49);post=np.arange(33,len(trial_start),dtype=np.int64);nv=int(np.ceil(len(post)*SPLIT_FRACTION));cut=int(post[-nv])
  # This is a continuous query timeline.  Split at a whole-trial boundary but
  # retain windows crossing trial boundaries inside either segment.  Only the
  # one train/validation boundary needs full-context isolation.  Starting a
  # validation window at cut_raw makes its endpoint cut_raw+49, so the first
  # 49 validation output bins are naturally not scored without dropping a
  # second 49-window block.
  padded=np.asarray(b.eligible_starts,np.int64);raw=padded-49;end=raw+50;cut_raw=int(trial_start[cut])
  ti=np.flatnonzero(end<=cut_raw);vi=np.flatnonzero(raw>=cut_raw)
  if not len(ti) or not len(vi):raise RuntimeError(f'{s}: empty whole-trial split')
  train[s]=QueryBank(s,b.X_store,np.asarray(b.target_store)[ti],padded[ti],b.unit_mask);val[s]=QueryBank(s,b.X_store,np.asarray(b.target_store)[vi],padded[vi],b.unit_mask)
  rows[s]={'nwb':str(nwb),'nwb_sha256':sha(nwb),'raw_trials':int(len(trial_start)),'post_m33_trials':int(len(post)),'validation_trial_count':nv,'cut_trial_id':cut,'cut_raw_bin':cut_raw,'train_trial_ids':[33,cut-1],'validation_trial_ids':[cut,int(post[-1])],'train_windows':int(len(ti)),'validation_windows':int(len(vi)),'validation_initial_endpoint_warmup_bins_not_scored':49,'train_eligible_starts_sha256':hashlib.sha256(np.asarray(padded[ti],np.int64).tobytes()).hexdigest(),'validation_eligible_starts_sha256':hashlib.sha256(np.asarray(padded[vi],np.int64).tobytes()).hexdigest(),'full_context_bins':50,'context_contained_within_segment':True,'cross_trial_windows_within_segment_allowed':True}
 lengths={s:len(train[s].eligible_starts) for s in old_plan.HELDIN_SESSIONS};manifest=sampler.build_shuffled_manifest(lengths,batch_size=BATCH,epochs=EPOCHS,sampler_seed=SPLIT_SEED)
 contract={'schema':'cross_session_m2_whole_trial_source_split_v1','source_surface':'source_train','selection_surface':'source_train_post_m33_last20pct_whole_trials','support_trials':list(range(33)),'validation_fraction':SPLIT_FRACTION,'rounding':'ceil last fraction of post-M33 whole trials','manifest':manifest,'sessions':rows}
 return train,val,tasks,contract
def ext4_query(device:torch.device):return query_surface('ext4',device)
