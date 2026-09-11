#!/usr/bin/env python3
"""Sealed all-24 EMA EXT6 selector for M2 fixed RIFT ablations.

It reads the ablation's public EXT6 bank only.  No EvalAI or hidden/test
record is opened.  Selection is the earliest maximum of complete six-session
EMA means and is deliberately separate from training.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Mapping
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];WS=ROOT.parent
for p in (ROOT,ROOT/'src',WS/'btransform_unified_v1'/'src',WS):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from scripts.m2_rift_r50_ablation_v1 import m2_ablation
from scripts.rift_v1 import m2_ext6_epoch_pick as shared
ARMS=('ACTIVITY_ONLY','NONE');SIX=tuple(shared.SIX)
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom(p:Path,x:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def digest(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def query_assets(q:Path,arm:str)->dict[str,Any]:
 r=json.loads((q/'official_heldout_query_banks.json').read_text())
 if r.get('schema')!='m2_rift_r50_ablation_ext6_query_v1' or r.get('arm')!=arm or r.get('hidden_or_test_opened') is not False:raise RuntimeError('unsealed/foreign EXT6 ablation bank')
 out={}
 for s in SIX:
  d=q/s;files=('mapping.json','e0_u.pt','T.npy','eligible_starts.npy','X_store.npy','target_store.npy')
  if not all((d/x).is_file() for x in files):raise RuntimeError(f'{s}: missing query asset')
  starts = np.load(d / 'eligible_starts.npy', mmap_mode='r')
  e0_obj = torch.load(d / 'e0_u.pt', map_location='cpu', weights_only=False)
  e0 = np.ascontiguousarray(e0_obj['E0'].detach().cpu().numpy(), np.float32)
  u = np.ascontiguousarray(e0_obj['frozen_u'].detach().cpu().numpy(), np.float32)
  carrier = np.ascontiguousarray(np.load(d / 'T.npy'), np.float32)
  expected = r.get('sessions', {}).get(s, {})
  if (int(expected.get('window_count', -1)) != len(starts) or
      hashlib.sha256(e0.tobytes()).hexdigest() != expected.get('E0_sha256') or
      hashlib.sha256(carrier.tobytes()).hexdigest() != expected.get('T_sha256') or
      hashlib.sha256(u.tobytes()).hexdigest() != expected.get('frozen_u_sha256') or
      list(u.shape) != expected.get('frozen_u_shape') or u.ndim != 3 or u.shape[1:] != (96, 64)):
   raise RuntimeError(f'{s}: query receipt array/window binding drift')
  if not np.array_equal(carrier, np.zeros((96, 4), np.float32)):
   raise RuntimeError(f'{s}: query direct T is not literal zero')
  if arm == 'NONE' and (not np.array_equal(e0, np.zeros((96, 50), np.float32)) or not np.array_equal(u, np.zeros_like(u))):
   raise RuntimeError(f'{s}: NONE query E0 is not literal zero')
  out[s]={'window_count':int(len(starts)),'files':{x:sha(d/x) for x in files}}
 return out
def check_meta(run:Path,arm:str)->dict[str,Any]:
 m=json.loads((run/'run_meta.json').read_text());r=json.loads((run/'train_receipt.json').read_text())
 if m.get('schema')!='m2_rift_r50_ablation_train_v1' or m.get('arm')!=arm or m.get('cell')!=m2_ablation.cell(arm):raise RuntimeError('run metadata arm/cell mismatch')
 if r.get('schema')!='m2_rift_r50_ablation_train_receipt_v1' or r.get('status')!='COMPLETED' or r.get('epochs')!=24 or r.get('global_step')!=75960:raise RuntimeError('requires completed 24x3165 formal train')
 if r.get('source_hashes')!=m.get('source_hashes') or r.get('frozen_cache_hashes')!=m.get('frozen_cache_hashes'):raise RuntimeError('receipt source/cache mismatch')
 for path,h in m['source_hashes'].items():
  if sha(Path(path))!=h:raise RuntimeError(f'training source drift {path}')
 return m
def report_ok(x:Mapping[str,Any],assets:Mapping[str,Any])->None:
 if x.get('partial') is not False or set(x.get('per_session',{}))!=set(SIX):raise RuntimeError('incomplete EXT6 report')
 if not math.isfinite(float(x.get('equal_session_mean',float('nan')))):raise RuntimeError('nonfinite mean')
 for s in SIX:
  z=x['per_session'][s]
  if int(z.get('window_count',-1))!=assets[s]['window_count'] or not math.isfinite(float(z.get('r2',float('nan')))):raise RuntimeError('EXT6 session drift')
def run(a:argparse.Namespace)->dict[str,Any]:
 run=a.run_dir.resolve();q=a.bank_cache_root.resolve()/'ext6_query';meta=check_meta(run,a.arm);assets=query_assets(q,a.arm);m2_ablation.patch(a.arm,a.bank_cache_root.resolve())
 dest=a.dest.resolve();dest.mkdir(parents=True,exist_ok=True);manifest={'schema':'m2_rift_r50_ablation_ext6_selection_v1','arm':a.arm,'run':str(run),'run_meta_sha256':sha(run/'run_meta.json'),'train_receipt_sha256':sha(run/'train_receipt.json'),'query_cache':str(q),'query_assets':assets,'selection_contract':'all 24 formal EMA checkpoints; six-session EXT6 unweighted equal-session mean; earliest maximum','official_test_used':False,'evalai_opened':False,'baseline_submission_id':582189}
 mp=dest/'manifest.json'
 if mp.exists():
  if json.loads(mp.read_text())!=manifest:raise RuntimeError('existing selection manifest drift')
 else:atom(mp,manifest)
 device=torch.device(a.device);torch.set_num_threads(a.cpu_threads);duals,banks=zip(*(shared.load_query_pair(s,q) for s in SIX));dm=dict(zip(SIX,duals));bm=dict(zip(SIX,banks));model=m2_ablation.base._decoder(device)
 pp=dest/'score_progress.json';progress=json.loads(pp.read_text()) if pp.exists() else {'schema':'m2_rift_r50_ablation_ext6_progress_v1','manifest_sha256':digest(manifest),'completed':{}}
 if progress.get('manifest_sha256')!=digest(manifest):raise RuntimeError('progress manifest mismatch')
 for epoch in range(1,25):
  cp=run/f'epoch_{epoch:03d}.pt';cs=sha(cp);prior=progress['completed'].get(str(epoch))
  if prior:
   if prior.get('checkpoint_sha256') != cs:
    raise RuntimeError('checkpoint changed')
   report_ok(prior, assets)
   continue
  st=torch.load(cp,map_location=device,weights_only=False)
  req={'schema':'m2_rift_concat_epoch_checkpoint_v1','cell':m2_ablation.cell(a.arm),'epoch':epoch,'global_step':epoch*3165,'smoke':False,'epochs':24,'context_bins':50,'attention_backend':'local','identity_interface':'concat'}
  if any(st.get(k)!=v for k,v in req.items()) or st.get('source_hashes')!=meta['source_hashes'] or st.get('frozen_cache_hashes')!=meta['frozen_cache_hashes']:raise RuntimeError(f'epoch {epoch} checkpoint contract mismatch')
  model.load_state_dict(st['raw_state_dict']);shadow=st['ema'].get('shadow',{});names=dict(model.named_parameters())
  if set(shadow)!=set(names):raise RuntimeError('EMA parameter keys drift')
  with torch.no_grad():
   for n,v in names.items():v.copy_(shadow[n].to(v.device,v.dtype))
  x=shared.score(model,dm,bm,device);report_ok(x,assets);progress['completed'][str(epoch)]={**x,'checkpoint_sha256':cs};atom(pp,progress)
 if set(progress['completed'])!={str(x) for x in range(1,25)}:raise RuntimeError('all24 required')
 vals={e:float(progress['completed'][str(e)]['equal_session_mean']) for e in range(1,25)};best=max(range(1,25),key=lambda e:(vals[e],-e));selected=progress['completed'][str(best)]
 # Package only decoder EMA; bank identity is separately sealed in the manifest.
 package=dest/'selected_ema.pt';st=torch.load(run/f'epoch_{best:03d}.pt',map_location='cpu',weights_only=False);model=m2_ablation.base._decoder(torch.device('cpu'));model.load_state_dict(st['raw_state_dict']);names=dict(model.named_parameters())
 with torch.no_grad():
  for n,v in names.items():v.copy_(st['ema']['shadow'][n].to(v.dtype))
 out={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 if package.exists():
  old=torch.load(package,map_location='cpu',weights_only=True)
  if set(old)!=set(out) or any(not torch.equal(old[k],out[k]) for k in out):raise RuntimeError('selected package differs')
 else:torch.save(out,package)
 receipt={'schema':'m2_rift_r50_ablation_ext6_selection_v1','status':'COMPLETED','arm':a.arm,'manifest_sha256':digest(manifest),'selection':{'rule':'earliest maximum finite unweighted equal_session_mean','epoch':best,'equal_session_mean':vals[best]},'selected':selected,'ema_by_epoch':progress['completed'],'selected_ema_state':{'path':str(package),'sha256':sha(package),'raw_state_serialized':False},'official_test_used':False,'evalai_opened':False,'utc':datetime.now(timezone.utc).isoformat()};atom(dest/'score_receipt.json',receipt);return receipt
def main()->int:
 p=argparse.ArgumentParser(description='score all 24 M2 ablation EMA checkpoints on sealed EXT6');p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--bank-cache-root',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=4);a=p.parse_args();print(json.dumps(run(a),indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
