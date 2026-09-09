#!/usr/bin/env python3
"""Seal a completed source-only B/D run before any target preparation."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
from protocol import SEED, STAGES, atomic_json, sha

def die(s): raise RuntimeError(s)
def read(p):
 if not p.is_file(): die(f'missing {p}')
 x=json.loads(p.read_text())
 if not isinstance(x,dict): die(f'object required: {p}')
 return x
def ah(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def bh(x): return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def roster(stage,dataset):
 row=STAGES[stage][dataset]
 if dataset=='h1':
  v1=str(Path(__file__).resolve().parents[3]/'btransform_unified_v1/src')
  if v1 not in sys.path:sys.path.insert(0,v1)
 if dataset=='m1':return list(row['sources']),list(row['targets'])
 from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
 ds=('1925-01-01','1925-01-08','1925-01-13','1925-01-15') if stage=='inner' else ('1925-01-01','1925-01-08','1925-01-13','1925-01-15','1925-01-19')
 return [s for d in ds for s in H1_SESSIONS_BY_DATE[d]],[s for d in row['targets'] for s in H1_SESSIONS_BY_DATE[d]]
def pack_info(path,dataset,stage,profile,sources,targets):
 if not path.is_file():die(f'missing carrier pack {path}')
 with np.load(path,allow_pickle=False) as z:
  if 'metadata' not in z.files:die('carrier metadata missing')
  meta=json.loads(str(z['metadata'].item()))
  if not isinstance(meta,dict) or any((meta.get(k)!=v) for k,v in {'schema':'carrier_last1_v2_pack','dataset':dataset,'stage':stage,'profile_id':profile,'surface':'source','sources':sources,'targets':[]}.items()):die('carrier pack identity/roster/surface drift')
  arrays={}
  for s in sources:
   key=f'carrier/{s}'
   if key not in z.files:die(f'carrier pack missing {key}')
   x=np.asarray(z[key],np.float32)
   if x.ndim!=2 or x.shape[1:]!=(4,) or not np.isfinite(x).all() or (dataset=='m1' and x.shape!=(64,4)):die(f'invalid carrier {key}')
   arrays[s]=ah(x)
 return {'path':str(path.resolve()),'sha256':sha(path),'metadata':meta,'array_sha256':arrays}
def pack_bytes(p,s):
 with np.load(p['path'],allow_pickle=False) as z:return bh(np.asarray(z[f'carrier/{s}'],np.float32))
def m1(run,stage,arm,pack):
 rec,pf=read(run/'train_receipt.json'),read(run/'preflight.json');src,tgt=roster(stage,'m1');split='m1_inner_last1_20120927' if stage=='inner' else 'm1_outer_last1_20120928'
 for row in (rec,pf):
  if any(row.get(k)!=v for k,v in {'split_id':split,'sources':src,'targets':tgt,'arm':arm,'seed':SEED}.items()):die('M1 receipt/preflight identity drift')
 if rec.get('status')!='COMPLETED' or pf.get('status')!='PASSED' or not pf.get('source_only') or rec.get('epochs')!=24 or not isinstance(rec.get('steps'),int) or rec['steps']<=0:die('M1 incomplete source train/preflight')
 if rec.get('selection',{}).get('target_query_labels_used') is not False:die('M1 target labels used for selection')
 if not isinstance(rec.get('batch_order_sha256'),str) or not isinstance(rec.get('initialization_hash'),str) or rec.get('actual_source_arrays')!=pf.get('source_evidence') or rec.get('data_contract')!=pf.get('data_contract'):die('M1 provenance drift')
 for name,digest in rec.get('source_code_sha256',{}).items():
  if not Path(name).is_file() or sha(Path(name))!=digest:die(f'M1 source code drift {name}')
 curve=[];cks={}
 for ep in range(1,25):
  jp,pt=run/f'ema_epoch_{ep:03d}.json',run/f'ema_epoch_{ep:03d}.pt';row=read(jp);vals=row.get('source_val',{}).get('per_session',{})
  if row.get('epoch')!=ep or not isinstance(row.get('step'),int) or set(vals)!=set(src) or not np.isfinite(list(vals.values())).all() or not pt.is_file():die(f'M1 EMA artifact drift e{ep}')
  curve.append(row);cks[jp.name]=sha(jp);cks[pt.name]=sha(pt)
 if any(curve[i]['step']<=curve[i-1]['step'] for i in range(1,24)) or curve[-1]['step']!=rec['steps']:die('M1 source step chain drift')
 selected=curve[0]
 for row in curve[1:]:
  if row['source_val']['equal_session_mean']>selected['source_val']['equal_session_mean']:selected=row
 if selected!=rec.get('selected'):die('M1 selection is not earliest source maximum')
 for n in ('resume_latest.pt','selected_ema.pt'):
  q=run/n
  if not q.is_file():die(f'M1 missing {q}')
  cks[n]=sha(q)
 if rec.get('checkpoint_sha256')!={n:cks[n] for n in ('resume_latest.pt','selected_ema.pt')}:die('M1 receipt checkpoint binding drift')
 evidence=rec['actual_source_arrays']
 if pack is not None and (set(evidence.get('carrier',{}))!=set(src) or any(evidence['carrier'][s]!=pack_bytes(pack,s) for s in src)):die('M1 train carrier does not bind sealed pack')
 return {'family':'m1','source_roster':src,'target_roster':tgt,'comparison':{'seed':SEED,'recipe':rec.get('recipe'),'data_contract':rec.get('data_contract'),'source_evidence':evidence,'initialization_hash':rec['initialization_hash'],'batch_order_sha256':rec['batch_order_sha256']},'selected':selected,'full_checkpoint_sha256':cks,'receipt':str((run/'train_receipt.json').resolve()),'receipt_sha256':sha(run/'train_receipt.json'),'preflight':str((run/'preflight.json').resolve()),'preflight_sha256':sha(run/'preflight.json')}
def h1_evidence(man):
 fields=('X_sha256','y_sha256','starts_sha256','endpoint_sha256','segment_starts_sha256','activity_sha256','support_trials','query_trials','stride')
 out={}
 for session,row in man.get('records',{}).items():
  val=row.get('validation',{})
  if not isinstance(val,dict) or any(k not in row for k in fields) or any(k not in val for k in fields):die(f'H1 source evidence field missing {session}')
  out[session]={'train':{k:row[k] for k in fields},'validation':{k:val[k] for k in fields}}
 return out
def h1(run,stage,profile,arm,pack):
 rec,pf,meta=read(run/'train_receipt.json'),read(run/'preflight.json'),read(run/'run_meta.json');src,tgt=roster(stage,'h1');split='h1_inner_last1_19250119' if stage=='inner' else 'h1_outer_last1_19250120'
 for row in (rec.get('meta',{}),meta,pf):
  x=row.get('split',row)
  if x.get('split_id')!=split or x.get('source_sessions')!=src or x.get('target_sessions')!=tgt:die('H1 split roster drift')
 if meta.get('arm')!=arm or meta.get('seed')!=SEED or pf.get('arm')!=arm or pf.get('status')!='PASSED' or not pf.get('source_only') or pf.get('target_records_opened')!=0:die('H1 arm/source preflight drift')
 if rec.get('schema')!='h1_carrier_last1_train_receipt_v1' or rec.get('recipe_binding')!=meta.get('recipe_binding'):die('H1 receipt binding drift')
 curve=rec.get('curve')
 if not isinstance(curve,list) or [x.get('epoch') for x in curve]!=list(range(1,33)) or any(not isinstance(x.get('step'),int) for x in curve) or any(curve[i]['step']<=curve[i-1]['step'] for i in range(1,32)):die('H1 full source curve/batch chain drift')
 selected=max(curve,key=lambda x:(x.get('source_val_ema',{}).get('equal_date_mean',float('-inf')),-x['epoch']))
 if rec.get('selected')!=selected:die('H1 selection is not earliest source equal-date maximum')
 cks=rec.get('checkpoints',{});want={f'ema_epoch_{i:03d}' for i in range(1,33)}
 if set(cks)!=want:die('H1 full e32 checkpoint authority missing')
 for n,d in cks.items():
  q=run/f'{n}.pt'
  if not q.is_file() or sha(q)!=d:die(f'H1 checkpoint SHA drift {n}')
 prepared=run.parents[2]/'prepared'/profile/arm/'source'/'manifest.json';man=read(prepared)
 if sha(prepared)!=meta.get('source_manifest_sha256') or sorted(man.get('records',{}))!=sorted(src):die('H1 source manifest recipe binding drift')
 auth=man.get('source_authority',{})
 if auth.get('split_id')!=split or auth.get('target_records_opened')!=0:die('H1 source authority leak/drift')
 clarification=None
 if stage=='outer':
  cp=prepared.parent.parent/'outer_authority_clarification.json';clarification=read(cp)
  if clarification.get('schema')!='carrier_last1_v2_outer_h1_authority_erratum_v1' or clarification.get('status')!='SEALED' or clarification.get('stage')!='outer' or clarification.get('dataset')!='h1' or clarification.get('prepared_root')!=str(prepared.parent.parent) or clarification.get('source_manifest_sha256')!=sha(prepared) or clarification.get('source_sessions')!=src or clarification.get('target_sessions')!=tgt or clarification.get('historical_text_correction_only') is not True or clarification.get('algorithm_changed') is not False or clarification.get('source_arrays_rewritten') is not False:die('outer H1 authority clarification binding drift')
  for key in ('source_hc_plan','source_hc_plan_arrays','adapter'):
   q=Path(clarification.get(key,''));digest=clarification.get(key+'_sha256')
   if not q.is_file() or sha(q)!=digest:die('outer H1 clarification SHA drift')
 for name,digest in meta.get('source_code_sha256',{}).items():
  if not Path(name).is_file() or sha(Path(name))!=digest:die(f'H1 source code drift {name}')
 records=man['records']
 if arm=='D_JOINT':
  if any('carrier_sha256' not in records[s] for s in src):die('H1 D lacks carrier evidence')
  if pack is not None and any(records[s]['carrier_sha256']!=pack['array_sha256'][s] for s in src):die('H1 prepared carrier does not bind pack')
 # Legacy H1 prepared files retain a carrier cache even for B; B loader zeros it.
 elif pack is not None:die('H1 B cannot bind a carrier pack')
 return {'family':'h1','source_roster':src,'target_roster':tgt,'comparison':{'seed':SEED,'recipe':{k:v for k,v in meta.items() if k not in {'arm','seed','schema','recipe_binding','source_manifest_sha256','initial_parameter_sha256'}},'source_authority':auth,'outer_authority_clarification':None if clarification is None else {'path':str((prepared.parent.parent/'outer_authority_clarification.json').resolve()),'sha256':sha(prepared.parent.parent/'outer_authority_clarification.json')},'source_authority_files':man.get('source_authority_files'),'source_evidence':h1_evidence(man),'initial_parameter_sha256':meta.get('initial_parameter_sha256'),'sampler_chain':[x.get('sampler_endpoint_keep_sha256') for x in curve]},'selected':selected,'full_checkpoint_sha256':cks,'receipt':str((run/'train_receipt.json').resolve()),'receipt_sha256':sha(run/'train_receipt.json'),'preflight':str((run/'preflight.json').resolve()),'preflight_sha256':sha(run/'preflight.json'),'source_manifest':str(prepared.resolve()),'source_manifest_sha256':sha(prepared)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',choices=('m1','h1'),required=True);p.add_argument('--stage',choices=('inner','outer'),required=True);p.add_argument('--profile-id',choices=('old','encoding','state','muscle'),required=True);p.add_argument('--arm',choices=('B_ACTIVITY_ONLY','D_JOINT'),required=True);p.add_argument('--carrier-pack',type=Path);a=p.parse_args();run=a.run.resolve();out=run/'source_seal.json'
 if out.exists():raise FileExistsError(out)
 if a.arm=='B_ACTIVITY_ONLY' and (a.profile_id!='old' or a.carrier_pack):die('B is one old-profile run with no carrier pack')
 old_literal=a.profile_id=='old' and a.arm=='D_JOINT'
 if a.arm=='D_JOINT' and not old_literal and not a.carrier_pack:die('new D requires a profile-specific carrier pack')
 if old_literal and a.carrier_pack:die('literal old-D cannot bind a new carrier pack')
 src,tgt=roster(a.stage,a.dataset);pack=None if not a.carrier_pack else pack_info(a.carrier_pack.resolve(),a.dataset,a.stage,a.profile_id,src,tgt);body=m1(run,a.stage,a.arm,pack) if a.dataset=='m1' else h1(run,a.stage,a.profile_id,a.arm,pack)
 atomic_json(out,{'schema':'carrier_last1_v2_source_seal','status':'SEALED','dataset':a.dataset,'stage':a.stage,'profile_id':a.profile_id,'arm':a.arm,'carrier_pack':pack,**body})
if __name__=='__main__':main()
