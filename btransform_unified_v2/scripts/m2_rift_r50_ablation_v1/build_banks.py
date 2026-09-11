#!/usr/bin/env python3
"""Build sealed M2 RIFT ablation banks without modifying the frozen cache."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2];WS=ROOT.parent
for p in (ROOT,ROOT/'src',WS/'btransform_unified_v1'/'src',WS):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from tfpd_exploration.src.m2_dual_track_v1 import champion,data,plan
ARMS=('ACTIVITY_ONLY','NONE'); BASE=WS/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'; QUERY=WS/'tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query'
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ash(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atom(p:Path,x:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def encoder():
 from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder
 x=HoldContrastFiLMEarlyPoolEncoder(100,50,64,side_dim=8,film_rank=8,num_post_layers=3,film_input='t4_plus_contrast'); champion.overlay_canonical_p0_and_empty_head(x);return x.eval()
def link(src:Path,dst:Path,names:tuple[str,...]):
 dst.mkdir(parents=True,exist_ok=True)
 for n in names: os.symlink((src/n).resolve(),dst/n)
def identity(activity: np.ndarray, arm: str, enc) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
 zt = np.zeros((96, 4), np.float32)
 if arm == 'NONE':
  u = np.zeros((activity.shape[0], 96, 64), np.float32)
  return np.zeros((96, 50), np.float32), zt, u, {'encoder_called': False, 'e0': 'literal_zero', 'frozen_u': 'literal_zero'}
 side = champion.empty_contrast_side(zt)
 e0, u = champion.native_e0_and_u(enc, torch.from_numpy(np.ascontiguousarray(activity, np.float32)), side)
 return np.ascontiguousarray(e0.numpy(), np.float32), zt, np.ascontiguousarray(u.numpy(), np.float32), {'encoder_called': True, 'e0_path': 'native reset_stream/push_trial/finalize_identity', 'side': 'empty_contrast_side(literal_zero_T)'}
def public_builder():
 src=WS/'tfpd_exploration/scripts/run_m2_small_s1_visible_ext6_epoch_pick_v1.py';sp=importlib.util.spec_from_file_location('m2_pub',src);m=importlib.util.module_from_spec(sp);assert sp and sp.loader;sp.loader.exec_module(m);return m,src
def main()->int:
 p=argparse.ArgumentParser(description='build isolated M2 fixed ablation cache and EXT6 query bank');p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args(); dest=a.dest.resolve()
 if dest.exists():raise FileExistsError('fresh destination required')
 if not BASE.is_dir() or not QUERY.is_dir():raise FileNotFoundError('frozen main cache/query cache missing')
 enc=None if a.arm=='NONE' else encoder(); rows={}; names=('X_store.npy','target_store.npy','eligible_starts.npy','calib_activity.npy','mapping.json','extra.json')
 for surface,sessions in (('source_train',plan.HELDIN_SESSIONS),('source_minival',plan.HELDIN_SESSIONS),('ext4',plan.EXT4_SESSIONS)):
  for s in sessions:
   src=BASE/surface/s;dst=dest/surface/s;link(src,dst,names);act=np.load(src/'calib_activity.npy');e0,t,u,proof=identity(act,a.arm,enc);np.save(dst/'T.npy',t);torch.save({'E0':torch.from_numpy(e0),'frozen_u':torch.from_numpy(u)},dst/'e0_u.pt');atom(dst/'provenance.json',{'arm':a.arm,'surface':surface,'session':s,'E0_sha256':ash(e0),'T_sha256':ash(t),'frozen_u_sha256':ash(u),'frozen_u_shape':list(u.shape),'source_X_sha256':sha(src/'X_store.npy'),'source_target_sha256':sha(src/'target_store.npy'),'source_starts_sha256':sha(src/'eligible_starts.npy'),**proof});rows[f'{surface}/{s}']={'E0_sha256':ash(e0),'T_sha256':ash(t),'frozen_u_sha256':ash(u),'frozen_u_shape':list(u.shape),'X_store_sha256':sha(src/'X_store.npy'),'target_store_sha256':sha(src/'target_store.npy'),'eligible_starts_sha256':sha(src/'eligible_starts.npy'),'calib_activity_sha256':sha(src/'calib_activity.npy'),**proof}
 # The exact same official public EXT6 builder provides only raw M33 calibration;
 # query X/y/starts/mapping remain byte-linked from the sealed query cache.
 builder,bsrc=public_builder();dataset,access=builder._build_official_heldout_dataset();qrows={}
 for s in plan.EXTERNAL_SESSIONS:
  src=QUERY/s;dst=dest/'ext6_query'/s;link(src,dst,('X_store.npy','target_store.npy','eligible_starts.npy','mapping.json'));bundle=data._calib_bundle(dataset,s);act=np.ascontiguousarray(bundle['activity'],np.float32)
  if act.shape!=(33,100,96):raise RuntimeError(f'{s}: M33 geometry {act.shape}')
  e0,t,u,proof=identity(act,a.arm,enc);np.save(dst/'T.npy',t);torch.save({'E0':torch.from_numpy(e0),'frozen_u':torch.from_numpy(u)},dst/'e0_u.pt');qrows[s]={'window_count':int(len(np.load(src/'eligible_starts.npy'))),'E0_sha256':ash(e0),'T_sha256':ash(t),'frozen_u_sha256':ash(u),'frozen_u_shape':list(u.shape),'X_link_sha256':sha(src/'X_store.npy'),'target_link_sha256':sha(src/'target_store.npy'),**proof}
 # FileAccessLog lets runtime reject hidden/test reads; builder owns the actual reader.
 if any(data.classify_nwb_role(x)=='hidden_or_test' for x in access.opened):raise RuntimeError('forbidden hidden/test opened')
 qreceipt={'schema':'m2_rift_r50_ablation_ext6_query_v1','status':'COMPLETED','arm':a.arm,'sessions':qrows,'opened_public_calib_nwbs':[str(x) for x in access.opened],'hidden_or_test_opened':False,'official_test_used':False,'query_source':str(QUERY),'builder':str(bsrc),'builder_sha256':sha(bsrc)};atom(dest/'ext6_query'/'official_heldout_query_banks.json',qreceipt)
 manifest={'schema':'m2_rift_r50_ablation_bank_v1','status':'COMPLETED','arm':a.arm,'input_contract':('ACTIVITY_ONLY: literal T[96,4]=0; frozen EMPTY-head encoder remelts E0 from M33 trialwise native reset/push/finalize.' if a.arm=='ACTIVITY_ONLY' else 'NONE: literal E0[96,50]=0 and literal T[96,4]=0; encoder not called.'),'source_manifest_contract':'24x3165=75960 seed42 R50D4 local13/12/12/12 batch32 AdamW3e-4 warmup1 EMA.9995 dropout.1','baseline_submission_id':582189,'baseline_official_ho':0.34654225938843214,'baseline_local_ext6_not_official':True,'base_cache':str(BASE),'base_cache_meta_sha256':sha(BASE/'meta.json'),'builder_sha256':sha(Path(__file__)),'rows':rows,'ext6_query_receipt':'ext6_query/official_heldout_query_banks.json','ext6_query_receipt_sha256':sha(dest/'ext6_query'/'official_heldout_query_banks.json'),'official_test_used':False,'evalai_opened':False,'created_utc':datetime.now(timezone.utc).isoformat()};atom(dest/'manifest.json',manifest);print(json.dumps(manifest,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
