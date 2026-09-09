#!/usr/bin/env python3
"""Fail-closed paired B / old-D / candidate-D audit.

``--phase source`` is the gate before target preparation.  ``--phase final``
replays every published score from immutable prediction artifacts and refuses
any source/target receipt that is not anchored to that gate.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from protocol import PROFILES, STAGES, atomic_json, sha

def die(s): raise RuntimeError(s)
def read(p):
 if not p.is_file():die(f'missing {p}')
 x=json.loads(p.read_text())
 if not isinstance(x,dict):die(f'object required {p}')
 return x
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def typed_sha(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def r2(y,p):
 # Historical scorer compatibility: all target elements share one float64 mean.
 y=np.asarray(y,np.float64);p=np.asarray(p,np.float64)
 if y.shape!=p.shape or y.ndim!=2 or not np.isfinite(y).all() or not np.isfinite(p).all():die('invalid target/prediction arrays')
 centered=y-y.mean();den=float(np.sum(centered*centered))
 if den<=0:die('zero target variance')
 return float(1.-np.sum((y-p)*(y-p))/den)
def channel_variance_weighted_r2(y,p):
 y=np.asarray(y,np.float64);p=np.asarray(p,np.float64)
 if y.shape!=p.shape or y.ndim!=2 or not np.isfinite(y).all() or not np.isfinite(p).all():die('invalid target/prediction arrays')
 den=float(np.var(y,axis=0).sum())
 if den<=0:die('zero target variance')
 return float(1.-np.sum(np.mean((y-p)**2,axis=0))/den)
def run_path(root,stage,dataset,profile,arm):return root/stage/dataset/profile/arm/'s42'
def seal(root,stage,dataset,profile,arm):
 p=run_path(root,stage,dataset,profile,arm)/'source_seal.json';x=read(p)
 if x.get('schema')!='carrier_last1_v2_source_seal' or x.get('status')!='SEALED' or any(x.get(k)!=v for k,v in {'stage':stage,'dataset':dataset,'profile_id':profile,'arm':arm}.items()):die(f'seal identity drift {p}')
 rec=Path(x.get('receipt',''))
 if not rec.is_file() or x.get('receipt_sha256')!=sha(rec):die(f'seal receipt SHA drift {p}')
 if not isinstance(x.get('full_checkpoint_sha256'),dict) or not x['full_checkpoint_sha256']:die(f'full checkpoint authority absent {p}')
 if arm=='B_ACTIVITY_ONLY':
  if x.get('carrier_pack') is not None:die('B must not bind a carrier pack')
 elif profile=='old':
  if x.get('carrier_pack') is not None:die('literal old-D must use only built-in carrier authority')
 else:
  q=x.get('carrier_pack') or {};path=Path(q.get('path',''))
  if not path.is_file() or q.get('sha256')!=sha(path):die(f'D carrier-pack binding drift {p}')
 return x,p
def h1_manifest_evidence(seal):
 man=read(Path(seal['source_manifest']))
 fields=('X_sha256','y_sha256','starts_sha256','endpoint_sha256','segment_starts_sha256','activity_sha256','support_trials','query_trials','stride')
 out={}
 for session,row in man.get('records',{}).items():
  val=row.get('validation',{})
  if not isinstance(val,dict) or any(k not in row for k in fields) or any(k not in val for k in fields):die(f'H1 source evidence field missing {session}')
  out[session]={'train':{k:row[k] for k in fields},'validation':{k:val[k] for k in fields}}
 return out
def source_rows(root,stage,dataset,profiles):
 rows={'B':seal(root,stage,dataset,'old','B_ACTIVITY_ONLY')}
 for profile in profiles:rows[f'{profile}_D']=seal(root,stage,dataset,profile,'D_JOINT')
 # common split and frozen source data/authority.  Carrier is the sole permitted
 # D difference, and B is deliberately activity-only.
 vals=[x[0] for x in rows.values()]
 if any(x['source_roster']!=vals[0]['source_roster'] or x['target_roster']!=vals[0]['target_roster'] for x in vals):die('paired roster drift')
 if dataset=='m1':
  ref=vals[0]['comparison'];recipe=ref.get('recipe');evidence=ref.get('source_evidence',{})
  if any(k not in evidence for k in ('calib','train','val','carrier','rsyn3_dictionary_and_normalizer')):die('M1 source evidence fields missing')
  base={k:evidence[k] for k in ('calib','train','val')};aux=evidence['rsyn3_dictionary_and_normalizer']
  for x in vals:
   c=x['comparison'];e=c.get('source_evidence',{})
   if any(k not in e for k in ('calib','train','val','carrier','rsyn3_dictionary_and_normalizer')):die('M1 source evidence fields missing')
   if c.get('seed')!=42 or c.get('recipe')!=recipe or c.get('data_contract')!=ref.get('data_contract') or {k:e[k] for k in ('calib','train','val')}!=base or e['rsyn3_dictionary_and_normalizer']!=aux:die('M1 paired source/optimization evidence drift')
   if c.get('initialization_hash')!=ref.get('initialization_hash') or c.get('batch_order_sha256')!=ref.get('batch_order_sha256'):die('M1 paired initialization/batch chain drift')
 else:
  ref=vals[0]['comparison'];recipe=ref.get('recipe');authority=ref.get('source_authority');files=ref.get('source_authority_files');initial=ref.get('initial_parameter_sha256');chain=ref.get('sampler_chain');evidence=h1_manifest_evidence(vals[0])
  for x in vals:
   c=x['comparison']
   if c.get('seed')!=42 or c.get('recipe')!=recipe or c.get('source_authority')!=authority or c.get('source_authority_files')!=files or h1_manifest_evidence(x)!=evidence:die('H1 paired source/optimization authority/evidence drift')
   if c.get('initial_parameter_sha256')!=initial or c.get('sampler_chain')!=chain:die('H1 paired initialization/batch chain drift')
 return rows
def source_audit(root,stage,dataset,profiles,dest):
 if dest.exists():raise FileExistsError(dest)
 rows=source_rows(root,stage,dataset,profiles)
 # A candidate must have a separately sealed D receipt and (where applicable)
 # a distinct pack; equality would silently collapse the ablation.
 d=[rows[f'{p}_D'][0] for p in profiles]
 pack_sha=[(x.get('carrier_pack') or {}).get('sha256') for x in d]
 meaningful=[x for x in pack_sha if x is not None]
 if len(set(meanful:=meaningful))!=len(meanful):die('candidate D packs reused')
 payload={'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'source','stage':stage,'dataset':dataset,'profiles':profiles,'rule':'source-only B / old-D / candidate-D seals, common source roster/init/optimizer/batches, before target preparation','source_seals':{k:{'path':str(p),'sha256':sha(p),'receipt_sha256':x['receipt_sha256'],'full_checkpoint_sha256':x['full_checkpoint_sha256']} for k,(x,p) in rows.items()},'comparison_sha256':stable({k:x['comparison'] for k,(x,_p) in rows.items()}),'outer_authority_clarifications':{k:x['comparison'].get('outer_authority_clarification') for k,(x,_p) in rows.items() if x['comparison'].get('outer_authority_clarification') is not None}}
 atomic_json(dest,payload);return payload
def m1_score(run,sealed):
 out=[]
 for target in sealed['target_roster']:
  d=run/f'score_{target}';rec=read(d/'score_receipt.json')
  if rec.get('status')!='COMPLETED' or rec.get('target')!=target or rec.get('target_labels_used_for_selection') is not False or rec.get('target_optimizer_steps')!=0:die(f'M1 target score policy drift {d}')
  if rec.get('checkpoint_sha256')!={k:v for k,v in sealed['full_checkpoint_sha256'].items() if k in ('resume_latest.pt','selected_ema.pt')}:die('M1 target score checkpoint binding drift')
  metrics=[];target_binding=None
  for key,file in (('target_selected_ema','target_selected_predictions.npz'),('target_epoch24_ema','target_epoch24_predictions.npz')):
   meta=rec.get('target_audit_arrays',{}).get('selected' if key=='target_selected_ema' else 'epoch24',{});p=d/file
   if not p.is_file() or meta.get('file')!=file or meta.get('sha256')!=sha(p):die(f'M1 target artifact SHA drift {p}')
   with np.load(p,allow_pickle=False) as z:
    if not {'target','prediction'}.issubset(z.files):die('M1 target arrays missing')
    if 'output_index_padded' not in z.files or 'window_start_padded' not in z.files or len(z['target'])!=len(z['output_index_padded']) or len(z['target'])!=len(z['window_start_padded']):die('M1 endpoint geometry drift')
    binding={k:typed_sha(z[k]) for k in ('target','window_start_padded','output_index_padded','output_index_query_relative','prefix_bins') if k in z.files}
    if set(binding)!={'target','window_start_padded','output_index_padded','output_index_query_relative','prefix_bins'}:die('M1 target binding keys missing')
    if target_binding is not None and binding!=target_binding:die('M1 selected/fixed target endpoint drift')
    target_binding=binding
    value=r2(z['target'],z['prediction']);channel_value=channel_variance_weighted_r2(z['target'],z['prediction'])
   reported=rec.get(key,{}).get('equal_session_mean')
   if not isinstance(reported,(int,float)) or not math.isclose(value,float(reported),rel_tol=0,abs_tol=1e-12):die(f'M1 numeric replay mismatch {p}')
   metrics.append({'kind':key,'r2':value,'artifact_sha256':sha(p),'channel_variance_weighted_r2':channel_value,'target_binding':binding})
  out.append({'target':target,'receipt_sha256':sha(d/'score_receipt.json'),'replay':metrics})
 return out
def h1_score(run,sealed):
 p=run/'target_score.json';rec=read(p)
 if rec.get('schema')!='h1_carrier_last1_target_score_v1' or rec.get('target_query_labels_used_for_gradients') is not False or rec.get('target_query_labels_used_for_selection') is not False:die('H1 target score policy drift')
 out=[];target_bindings={}
 for key in ('selected_ema','fixed_e32_ema'):
  item=rec.get(key,{})
  if item.get('checkpoint_sha256')!=sealed['full_checkpoint_sha256'].get(f'ema_epoch_{int(item.get("epoch",-1)):03d}'):die('H1 target checkpoint binding drift')
  target=item.get('target',{});per=target.get('per_session',{})
  values=[];channel_values=[]
  for session,row in per.items():
   artifact=Path(row.get('artifact',''))
   if not artifact.is_file() or row.get('artifact_sha256')!=sha(artifact):die(f'H1 target artifact SHA drift {artifact}')
   with np.load(artifact,allow_pickle=False) as z:
    if 'endpoints' not in z.files or len(z['target'])!=len(z['endpoints']):die('H1 endpoint geometry drift')
    if row.get('endpoint_sha256') and hashlib.sha256(np.ascontiguousarray(z['endpoints']).dtype.str.encode()+str(np.asarray(z['endpoints']).shape).encode()+np.ascontiguousarray(z['endpoints']).tobytes()).hexdigest()!=row['endpoint_sha256']:die('H1 endpoint SHA drift')
    binding={'target':typed_sha(z['target']),'endpoints':typed_sha(z['endpoints'])}
    if session in target_bindings and target_bindings[session]!=binding:die('H1 selected/fixed target endpoint drift')
    target_bindings[session]=binding
    value=r2(z['target'],z['prediction']);channel_value=channel_variance_weighted_r2(z['target'],z['prediction'])
   if not math.isclose(value,float(row.get('r2',float('nan'))),rel_tol=0,abs_tol=1e-12):die(f'H1 numeric replay mismatch {artifact}')
   values.append(value);channel_values.append(channel_value)
  if set(per)!=set(sealed['target_roster']) or not math.isclose(float(np.mean(values)),float(target.get('equal_session_mean',float('nan'))),rel_tol=0,abs_tol=1e-12):die('H1 target aggregate replay mismatch')
  out.append({'kind':key,'epoch':item['epoch'],'per_session_r2':{session: float(per[session]['r2']) for session in per},'per_session_channel_variance_weighted_r2':{session: channel_values[i] for i,session in enumerate(per)},'sessionmean_channel_variance_weighted_r2':float(np.mean(channel_values)),'target_bindings':dict(target_bindings),'checkpoint_sha256':item['checkpoint_sha256']})
 return {'receipt_sha256':sha(p),'replay':out}
def verify_final_gate(g,rows,stage,dataset,profiles):
 if any(g.get(k)!=v for k,v in {'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'source','stage':stage,'dataset':dataset,'profiles':profiles}.items()):die('final source gate identity drift')
 expected={k:{'path':str(p),'sha256':sha(p),'receipt_sha256':x['receipt_sha256'],'full_checkpoint_sha256':x['full_checkpoint_sha256']} for k,(x,p) in rows.items()}
 if g.get('source_seals')!=expected:die('final source gate seal binding drift')
 for x,p in rows.values():
  # Revalidate every receipt, preflight, source manifest and checkpoint file recorded by the live seal.
  for key,hashkey in (('receipt','receipt_sha256'),('preflight','preflight_sha256'),('source_manifest','source_manifest_sha256')):
   if x.get(key) is not None:
    q=Path(x[key])
    if not q.is_file() or x.get(hashkey)!=sha(q):die(f'live source artifact SHA drift: {key}')
  for name,digest in x['full_checkpoint_sha256'].items():
   q=Path(x['receipt']).parent/(name if Path(name).suffix else name+'.pt')
   if not q.is_file() or sha(q)!=digest:die(f'live checkpoint SHA drift: {name}')
  for name,digest in x.get('comparison',{}).get('recipe',{}).get('source_code_sha256',{}).items():
   q=Path(name)
   if not q.is_file() or sha(q)!=digest:die('live source code SHA drift')
def cross_target_bindings(dataset,reported):
 if dataset=='m1':
  base=reported.get('B')
  if not isinstance(base,list):die('M1 B target replay missing')
  names={row.get('target') for row in base}
  if None in names or len(names)!=len(base):die('M1 B target roster malformed')
  for arm,rows in reported.items():
   actual={row.get('target') for row in rows} if isinstance(rows,list) else set()
   if actual!=names or len(rows)!=len(names):die('M1 cross-arm target roster drift')
  for base_row in base:
   target_name=base_row['target'];ref={m['kind']:m['target_binding'] for m in base_row['replay']}
   for arm,rows in reported.items():
    item=next(row for row in rows if row['target']==target_name)
    if {m['kind']:m['target_binding'] for m in item['replay']}!=ref:die('M1 cross-arm target/endpoint binding drift')
 else:
  base=reported.get('B',{});ref={x['kind']:x['target_bindings'] for x in base.get('replay',[])}
  if set(ref)!={'selected_ema','fixed_e32_ema'}:die('H1 B target replay malformed')
  for arm,row in reported.items():
   actual={x['kind']:x['target_bindings'] for x in row.get('replay',[])} if isinstance(row,dict) else {}
   if set(actual)!=set(ref) or actual!=ref:die('H1 cross-arm target/endpoint binding drift')
def final_audit(root,stage,dataset,profiles,dest):
 gate=root/stage/dataset/'source_gate.json';g=read(gate)
 rows=source_rows(root,stage,dataset,profiles);verify_final_gate(g,rows,stage,dataset,profiles);reported={}
 for key,(x,_p) in rows.items():
  profile,arm=('old','B_ACTIVITY_ONLY') if key=='B' else (key[:-2],'D_JOINT');run=run_path(root,stage,dataset,profile,arm)
  reported[key]=m1_score(run,x) if dataset=='m1' else h1_score(run,x)
 cross_target_bindings(dataset,reported)
 if dest.exists():raise FileExistsError(dest)
 payload={'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'final','stage':stage,'dataset':dataset,'profiles':profiles,'source_gate_sha256':sha(gate),'rule':'target receipts/artifacts are source-gate-bound; numeric R2 replay only, no target selection or optimizer steps','metric_definitions':{'r2':'legacy_flattened_float64_global_mean: 1 - sum((y-p)^2)/sum((y-mean(y))^2); used for historical scorer compatibility only','channel_variance_weighted_r2':'diagnostic per-channel-centered variance-weighted R2: 1 - sum(mean((y-p)^2,axis=0))/sum(var(y,axis=0))'},'target_replay':reported}
 atomic_json(dest,payload);return payload
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--dataset',choices=('m1','h1'),required=True);p.add_argument('--stage',choices=('inner','outer'),required=True);p.add_argument('--profiles',nargs='+',choices=PROFILES,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--phase',choices=('source','final'),default='source');a=p.parse_args()
 if a.profiles[0]!='old' or len(set(a.profiles))!=len(a.profiles):die('profiles must start with one old profile and contain no duplicates')
 root=a.run_root.resolve();(source_audit if a.phase=='source' else final_audit)(root,a.stage,a.dataset,a.profiles,a.dest.resolve())
if __name__=='__main__':main()
