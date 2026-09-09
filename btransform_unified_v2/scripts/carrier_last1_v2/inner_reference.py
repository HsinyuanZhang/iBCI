#!/usr/bin/env python3
"""Bind a new inner candidate to the completed chronological-last2 B/old-D runs.

This is a read-only historical reference bridge.  It never manufactures a new
old receipt and it intentionally never reads the old outer target rows.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'results/chronological_last2_v1'
def die(s):raise RuntimeError(s)
def read(p):
 if not p.is_file():die(f'missing {p}')
 x=json.loads(p.read_text())
 if not isinstance(x,dict):die(f'object required {p}')
 return x
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def fresh(p):
 if p.exists():raise FileExistsError(p)
def metric_arrays(y,p):
 y=np.asarray(y,np.float64);p=np.asarray(p,np.float64)
 if y.shape!=p.shape or y.ndim!=2 or not np.isfinite(y).all() or not np.isfinite(p).all():die('invalid prediction artifact')
 return y,p
def legacy_flattened_r2(y,p):
 """Historical scorer-compatible R2: flatten output channels before centering."""
 y,p=metric_arrays(y,p);den=float(np.sum((y-y.mean())**2))
 if den<=0:die('zero flattened target variance')
 return float(1-np.sum((y-p)**2)/den)
def channel_variance_weighted_r2(y,p):
 """Per-output centered variance-weighted R2, with float64 arithmetic."""
 y,p=metric_arrays(y,p)
 den=float(np.var(y,axis=0).sum())
 if den<=0:die('zero target variance')
 return float(1-np.mean((y-p)**2,axis=0).sum()/den)
METRIC_FORMULAS={
 'target_replay':{'label':'legacy_flattened_r2','formula':'float64 1 - sum((y - prediction)^2) / sum((y - mean(all_elements(y)))^2); historical scorer-compatible'},
 'channelwise_replay':{'label':'channel_variance_weighted_r2','formula':'float64 1 - sum(mean((y - prediction)^2, axis=0)) / sum(var(y, axis=0)); each output channel is centered separately'},
}
def m1_receipt(run,arm):
 rec,pf=read(run/'train_receipt.json'),read(run/'preflight.json')
 if rec.get('status')!='COMPLETED' or pf.get('status')!='PASSED' or rec.get('arm')!=arm or pf.get('arm')!=arm:die('M1 incomplete/arm-drift receipt')
 if rec.get('sources')!=['ses-20120924','ses-20120926'] or rec.get('epochs')!=24 or not pf.get('source_only'):die('M1 inner source/e24 drift')
 allcp={}
 for e in range(1,25):
  j,q=run/f'ema_epoch_{e:03d}.json',run/f'ema_epoch_{e:03d}.pt';row=read(j)
  if row.get('epoch')!=e or not q.is_file():die(f'M1 missing fixed checkpoint {e}')
  allcp[j.name]=sha(j);allcp[q.name]=sha(q)
 for n in ('resume_latest.pt','selected_ema.pt'):
  q=run/n
  if not q.is_file():die(f'M1 missing {n}')
  allcp[n]=sha(q)
 if rec.get('checkpoint_sha256') and any(rec['checkpoint_sha256'].get(k)!=allcp[k] for k in rec['checkpoint_sha256']):die('M1 receipt checkpoint SHA drift')
 return rec,pf,allcp
def h1_receipt(run,prepared,arm):
 rec,pf=read(run/'train_receipt.json'),read(run/'preflight.json');meta=read(run/'run_meta.json')
 split=meta.get('split',rec.get('meta',{}).get('split',{}))
 if meta.get('arm')!=arm or pf.get('arm')!=arm or pf.get('status')!='PASSED' or not pf.get('source_only') or split.get('source_dates')!=['1925-01-01','1925-01-08','1925-01-13','1925-01-15']:die('H1 inner source/preflight drift')
 curve=rec.get('curve',[])
 if [x.get('epoch') for x in curve]!=list(range(1,33)):die('H1 fixed-e32 curve missing')
 cps=rec.get('checkpoints',{})
 if set(cps)!={f'ema_epoch_{e:03d}' for e in range(1,33)}:die('H1 all fixed checkpoints missing')
 for n,d in cps.items():
  q=run/f'{n}.pt'
  if not q.is_file() or sha(q)!=d:die(f'H1 checkpoint SHA drift {n}')
 man=read(prepared/'source'/'manifest.json')
 if meta.get('source_manifest_sha256') and meta['source_manifest_sha256']!=sha(prepared/'source'/'manifest.json'):die('H1 source manifest binding drift')
 return rec,pf,meta,man,cps
def m1_common_evidence(x):
 if not isinstance(x,dict) or any(k not in x for k in ('calib','train','val')):die('M1 common source evidence missing')
 return {k:x[k] for k in ('calib','train','val')}
def m1_carrier_authority(x):
 if not isinstance(x,dict) or any(k not in x for k in ('carrier','rsyn3_dictionary_and_normalizer')):die('M1 carrier-family authority missing')
 return {k:x[k] for k in ('carrier','rsyn3_dictionary_and_normalizer')}
def profile_from_pack(pack,dataset):
 with np.load(pack,allow_pickle=False) as z:
  if 'metadata' not in z.files:die('candidate pack metadata missing')
  meta=json.loads(str(z['metadata'].item()))
  if not isinstance(meta,dict) or meta.get('profile_id') not in {'state','muscle'} or meta.get('dataset')!=dataset or meta.get('stage')!='inner' or meta.get('surface')!='source' or meta.get('targets')!=[]:die('candidate pack identity/scope drift')
  sources=['ses-20120924','ses-20120926'] if dataset=='m1' else ['ses-19250101T111740','ses-19250101T112404','ses-19250108T110520','ses-19250108T111022','ses-19250108T111455','ses-19250113T120811','ses-19250113T121303','ses-19250115T110633','ses-19250115T111328']
  if meta.get('sources')!=sources:die('candidate pack source roster drift')
  arrays={}
  for session in sources:
   key=f'carrier/{session}'
   if key not in z.files:die(f'candidate pack missing {key}')
   x=np.ascontiguousarray(np.asarray(z[key],np.float32));arrays[session]={'bytes':hashlib.sha256(x.view(np.uint8)).hexdigest(),'typed':hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()}
 return meta['profile_id'],meta,arrays
def h1_evidence(man):
 fields=('X_sha256','y_sha256','starts_sha256','endpoint_sha256','segment_starts_sha256','activity_sha256','support_trials','query_trials','stride')
 out={}
 for session,row in man.get('records',{}).items():
  val=row.get('validation',{})
  if not isinstance(val,dict) or any(k not in row for k in fields) or any(k not in val for k in fields):die(f'H1 source evidence field missing {session}')
  out[session]={'train':{k:row[k] for k in fields},'validation':{k:val[k] for k in fields}}
 return out
def source(dataset,candidate,cprep,pack,out):
 fresh(out);profile,packmeta,packarrays=profile_from_pack(pack,dataset);old_audit=read(OLD/f'{dataset}_paired_audit.json')
 # Historical audit must itself be a completed audit. H1 has no status field.
 if dataset=='m1':
  if old_audit.get('schema')!='m1_chronological_last2_paired_audit_v1' or old_audit.get('status')!='PASSED':die('old M1 paired audit not passed')
  oldb,oldp,oldbc=m1_receipt(OLD/'m1/B_ACTIVITY_ONLY/s42','B_ACTIVITY_ONLY');oldd,oldq,olddc=m1_receipt(OLD/'m1/D_JOINT/s42','D_JOINT');new,newp,newc=m1_receipt(candidate,'D_JOINT')
  for r in (oldb,oldd,new):
   if r.get('seed')!=42:die('M1 seed drift')
  base=m1_common_evidence(oldb.get('actual_source_arrays',{}))
  if m1_common_evidence(oldd.get('actual_source_arrays',{}))!=base or m1_common_evidence(new.get('actual_source_arrays',{}))!=base:die('M1 source neural/activity/calibration evidence drift')
  carrier_authorities={'historical_B':m1_carrier_authority(oldb['actual_source_arrays']),'historical_old_D':m1_carrier_authority(oldd['actual_source_arrays']),'candidate_D':m1_carrier_authority(new['actual_source_arrays'])}
  if oldb.get('recipe')!=oldd.get('recipe') or oldb.get('recipe')!=new.get('recipe'):die('M1 actual optimization recipe drift')
  if oldb.get('initialization_hash')!=oldd.get('initialization_hash') or oldb.get('initialization_hash')!=new.get('initialization_hash') or oldb.get('batch_order_sha256')!=oldd.get('batch_order_sha256') or oldb.get('batch_order_sha256')!=new.get('batch_order_sha256') or oldp.get('checks',{}).get('counts')!=oldq.get('checks',{}).get('counts') or oldp.get('checks',{}).get('counts')!=newp.get('checks',{}).get('counts'):die('M1 common parameter/init/batch/dropout drift')
  if not pack or not pack.is_file() or any(new['actual_source_arrays']['carrier'].get(k)!=packarrays[k]['bytes'] for k in packarrays):die('candidate M1 pack/source bytes drift')
  refs={'B':(OLD/'m1/B_ACTIVITY_ONLY/s42',oldbc),'old_D':(OLD/'m1/D_JOINT/s42',olddc),'candidate_D':(candidate,newc)}
  comparison={'common_source_evidence':base,'carrier_family_authority':carrier_authorities,'recipe':oldb.get('recipe'),'initialization_hash':oldb.get('initialization_hash'),'batch_order_sha256':oldb.get('batch_order_sha256')}
 else:
  if old_audit.get('schema')!='h1_chronological_last2_audit_v1':die('old H1 paired audit unavailable')
  oldprep=OLD/'h1/prepared';oldb,oldp,oldm,oldman,oldbc=h1_receipt(OLD/'h1/B_ACTIVITY_ONLY',oldprep,'B_ACTIVITY_ONLY');oldd,oldq,olddm,olddman,olddc=h1_receipt(OLD/'h1/D_JOINT',oldprep,'D_JOINT');new,newp,newm,newman,newc=h1_receipt(candidate,cprep,'D_JOINT')
  if h1_evidence(oldman)!=h1_evidence(olddman) or h1_evidence(oldman)!=h1_evidence(newman):die('H1 source neural/activity/calibration evidence drift')
  def recipe(m):return {k:m.get(k) for k in ('seed','epochs','batch')}
  if recipe(oldm)!=recipe(olddm) or recipe(oldm)!=recipe(newm) or oldm.get('initial_parameter_sha256')!=olddm.get('initial_parameter_sha256') or oldm.get('initial_parameter_sha256')!=newm.get('initial_parameter_sha256'):die('H1 optimizer hyperparameter/init drift')
  # preflight proves B/D byte-equal shared decoder and B/D encoder initial state;
  # candidate must make the same concrete claims, not just carry a new schema.
  for pf in (oldp,oldq,newp):
   if not pf.get('shared_decoder_initial_states_byte_equal') or not pf.get('b_d_encoder_initial_state_equal') or not all(pf.get('gradient_checks',{}).values()) or pf.get('trainable_parameter_counts')!=oldp.get('trainable_parameter_counts'):die('H1 common parameter/init/preflight drift')
  if [x.get('sampler_endpoint_keep_sha256') for x in oldb['curve']]!=[x.get('sampler_endpoint_keep_sha256') for x in oldd['curve']] or [x.get('sampler_endpoint_keep_sha256') for x in oldb['curve']]!=[x.get('sampler_endpoint_keep_sha256') for x in new['curve']]:die('H1 batch/dropout chain drift')
  if not pack or not pack.is_file() or any(newman['records'][k].get('carrier_sha256')!=packarrays[k]['typed'] for k in packarrays):die('candidate H1 pack/source bytes drift')
  refs={'B':(OLD/'h1/B_ACTIVITY_ONLY',oldbc),'old_D':(OLD/'h1/D_JOINT',olddc),'candidate_D':(candidate,newc)};comparison={'source_evidence':h1_evidence(oldman),'optimizer_hyperparameters':recipe(oldm),'optimizer_code_sha256':{'historical_B':oldm.get('source_code_sha256'),'historical_old_D':olddm.get('source_code_sha256'),'candidate_D':newm.get('source_code_sha256')},'codepath_scope':'source-code hashes are bound for review; cross-schema paths are intentionally not equated','sampler_chain':[x.get('sampler_endpoint_keep_sha256') for x in oldb['curve']]}
 original={str(OLD/f'{dataset}_paired_audit.json'):sha(OLD/f'{dataset}_paired_audit.json'),str(pack.resolve()):sha(pack)}
 for name,(run,cps) in refs.items():
  original[str(run/'train_receipt.json')]=sha(run/'train_receipt.json');original[str(run/'preflight.json')]=sha(run/'preflight.json')
  for n,d in cps.items():original[str(run/(n if n.endswith('.pt') or n.endswith('.json') else n+'.pt'))]=d
 payload={'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'source','stage':'inner','dataset':dataset,'profiles':['old',profile],'historical_inner_reference':True,'purpose':'reuse completed chronological-last2 B and old-D as immutable inner references; candidate-D is newly trained','historical_original_bases':[{'path':p,'sha256':d} for p,d in sorted(original.items())],'original_bases_sha256':original,'source_seals':{k:{'run':str(v[0].resolve()),'fixed_checkpoints':v[1]} for k,v in refs.items()},'comparison':comparison,'candidate_pack':{'path':str(pack.resolve()),'sha256':sha(pack),'metadata':packmeta},'source_roster':comparison.get('common_source_evidence',comparison.get('source_evidence',{}))}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
def artifact(path,reported):
 if not path.is_file() or sha(path)!=reported:die(f'artifact SHA drift {path}')
 with np.load(path,allow_pickle=False) as z:return {'legacy_flattened_r2':legacy_flattened_r2(z['target'],z['prediction']),'channel_variance_weighted_r2':channel_variance_weighted_r2(z['target'],z['prediction'])}
def typed(x):
 a=np.ascontiguousarray(np.asarray(x));return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def m1_target_binding(path,reported):
 """Bind all M1 target labels and coordinate arrays, not only replayed R2."""
 if not path.is_file() or sha(path)!=reported:die(f'M1 artifact SHA drift {path}')
 required=('target','prediction','window_start_padded','output_index_padded','output_index_query_relative','prefix_bins')
 with np.load(path,allow_pickle=False) as z:
  if not set(required).issubset(z.files):die(f'M1 target artifact keys drift {path}')
  target,pred=z['target'],z['prediction']
  if target.shape!=pred.shape or target.ndim!=2:die('M1 target/prediction geometry drift')
  n=target.shape[0]
  if any(np.asarray(z[k]).ndim!=1 or (k!='prefix_bins' and len(z[k])!=n) for k in required[2:]):die('M1 target coordinate geometry drift')
  binding={k:typed(z[k]) for k in required if k!='prediction'}
  values={'legacy_flattened_r2':legacy_flattened_r2(target,pred),'channel_variance_weighted_r2':channel_variance_weighted_r2(target,pred)}
 return values,binding
def h1_target_binding(path,reported,row):
 """Bind typed target and endpoint arrays to receipt hashes and exact NPZ keys."""
 if not path.is_file() or sha(path)!=reported:die(f'H1 target artifact SHA drift {path}')
 with np.load(path,allow_pickle=False) as z:
  if set(z.files)!={'prediction','target','endpoints'}:die(f'H1 target artifact keys drift {path}')
  if z['target'].shape!=z['prediction'].shape or z['target'].ndim!=2 or len(z['target'])!=len(z['endpoints']):die('H1 target geometry drift')
  yhash,ehash=typed(z['target']),typed(z['endpoints'])
  if not isinstance(row.get('y_sha256'),str) or not isinstance(row.get('endpoint_sha256'),str) or yhash!=row['y_sha256'] or ehash!=row['endpoint_sha256']:die('H1 receipt typed target/endpoint hash drift')
  values={'legacy_flattened_r2':legacy_flattened_r2(z['target'],z['prediction']),'channel_variance_weighted_r2':channel_variance_weighted_r2(z['target'],z['prediction'])}
 return values,{'target':yhash,'endpoints':ehash}
def final(dataset,candidate,cprep,pack,out):
 fresh(out);profile,packmeta,_packarrays=profile_from_pack(pack,dataset)
 # source gate must be supplied as --out only after source; infer sibling source_gate
 gate=out.parent/'source_gate.json';g=read(gate)
 if g.get('schema')!='carrier_last1_v2_paired_audit' or g.get('status')!='PASSED' or g.get('stage')!='inner' or g.get('dataset')!=dataset or g.get('profiles')!=['old',profile] or not g.get('historical_inner_reference') or g.get('candidate_pack',{}).get('path')!=str(pack.resolve()) or g.get('candidate_pack',{}).get('sha256')!=sha(pack):die('historical inner source gate binding required')
 expected={'B':str((OLD/dataset/('B_ACTIVITY_ONLY/s42' if dataset=='m1' else 'B_ACTIVITY_ONLY')).resolve()),'old_D':str((OLD/dataset/('D_JOINT/s42' if dataset=='m1' else 'D_JOINT')).resolve()),'candidate_D':str(candidate.resolve())}
 if {k:v.get('run') for k,v in g.get('source_seals',{}).items()}!=expected:die('source gate run-path binding drift')
 for item in g.get('historical_original_bases',[]):
  q=Path(item.get('path',''))
  if not q.is_file() or item.get('sha256')!=sha(q):die('source gate original base SHA drift')
 rows={};channel_rows={}
 if dataset=='m1':
  target='ses-20120927';target_signatures=None;runs={'B':OLD/'m1/B_ACTIVITY_ONLY/s42','old_D':OLD/'m1/D_JOINT/s42','candidate_D':candidate}
  for name,run in runs.items():
   rec=read(run/f'score_{target}'/'score_receipt.json')
   if rec.get('target')!=target or rec.get('target_optimizer_steps')!=0 or rec.get('target_labels_used_for_selection') is not False:die('M1 target update/selection drift')
   vals={};channel_vals={};bindings={}
   for key,sub in [('fixed_e24','epoch24'),('selected','selected')]:
    q=run/f'score_{target}/target_{sub if sub=="selected" else "epoch24"}_predictions.npz';meta=rec['target_audit_arrays']['selected' if sub=='selected' else 'epoch24']
    metrics,binding=m1_target_binding(q,meta['sha256']);legacy=metrics['legacy_flattened_r2']
    if not math.isclose(legacy,float(rec['target_selected_ema' if sub=='selected' else 'target_epoch24_ema']['equal_session_mean']),abs_tol=1e-12):die('M1 historical flattened R2 replay drift')
    vals[key]=legacy
    channel_vals[key]=metrics['channel_variance_weighted_r2']
    bindings[key]=binding
   if bindings['fixed_e24']!=bindings['selected']:die('M1 selected/fixed target or endpoint drift')
   rows[name]=vals;channel_rows[name]=channel_vals;target_signatures=bindings if name=='B' else target_signatures
   if name!='B' and bindings!=target_signatures:die('M1 target/y/coordinate arrays differ from historical B')
 else:
  date='19250119';target_signatures={};runs={'B':OLD/'h1/B_ACTIVITY_ONLY','old_D':OLD/'h1/D_JOINT','candidate_D':candidate}
  for name,run in runs.items():
   rec=read(run/'target_score.json')
   if rec.get('target_query_labels_used_for_gradients') is not False or rec.get('target_query_labels_used_for_selection') is not False:die('H1 target update/selection drift')
   vals={};channel_vals={}
   for key,field in [('fixed_e32','fixed_e32_ema'),('selected','selected_ema')]:
    per=rec[field]['target']['per_session'];use=[(s,v) for s,v in per.items() if date in s]
    if len(use)!=2:die('H1 inner date must contain exactly two target sessions')
    scores=[]
    for session,v in use:
     metrics,binding=h1_target_binding(Path(v['artifact']),v['artifact_sha256'],v);legacy=metrics['legacy_flattened_r2']
     if not math.isclose(legacy,float(v.get('r2',float('nan'))),abs_tol=1e-12):die('H1 historical flattened R2 replay drift')
     scores.append((session,legacy,metrics['channel_variance_weighted_r2'],binding['target'],binding['endpoints']))
    if key=='fixed_e32': anchor=scores
    elif [(x[0],x[3],x[4]) for x in scores] != [(x[0],x[3],x[4]) for x in anchor]:die('H1 selected/fixed target endpoint drift')
    vals[key]=float(np.mean([x[1] for x in scores]))
    channel_vals[key]={'per_session':{x[0]:x[2] for x in scores},'equal_session_mean':float(np.mean([x[2] for x in scores]))}
    sig=[(x[0],x[3],x[4]) for x in scores]
    if key=='fixed_e32': target_signatures[name]=sig
   rows[name]=vals;channel_rows[name]=channel_vals
  if set(target_signatures)!=set(runs):die('H1 target signatures missing arm')
  for arm,sig in target_signatures.items():
   if sig!=target_signatures['B']:die(f'H1 target endpoints/y differ from historical B: {arm}')
 primary='fixed_e24' if dataset=='m1' else 'fixed_e32'
 channel_fixed={name:(channel_rows[name][primary] if dataset=='m1' else channel_rows[name][primary]['equal_session_mean']) for name in rows}
 payload={'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'final','stage':'inner','dataset':dataset,'profiles':['old',profile],'historical_inner_reference':True,'source_gate_sha256':sha(gate),'inner_target_only':'ses-20120927' if dataset=='m1' else '1925-01-19','metric_formula':METRIC_FORMULAS,'target_replay':rows,'channelwise_replay':channel_rows,'target_typed_bindings':target_signatures,'fixed_primary_deltas':{'candidate_minus_B':rows['candidate_D'][primary]-rows['B'][primary],'candidate_minus_oldD':rows['candidate_D'][primary]-rows['old_D'][primary]},'channelwise_fixed_deltas':{'candidate_minus_B':channel_fixed['candidate_D']-channel_fixed['B'],'candidate_minus_oldD':channel_fixed['candidate_D']-channel_fixed['old_D']}}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('m1','h1'),required=True);p.add_argument('--candidate-run',type=Path,required=True);p.add_argument('--candidate-prepared',type=Path);p.add_argument('--candidate-pack',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=('source','final'),default='source');a=p.parse_args();a.candidate_run=a.candidate_run.resolve();a.candidate_pack=a.candidate_pack.resolve();a.out=a.out.resolve();a.candidate_prepared=None if a.candidate_prepared is None else a.candidate_prepared.resolve()
 if a.dataset=='h1' and a.candidate_prepared is None:die('--candidate-prepared required for H1')
 (source if a.phase=='source' else final)(a.dataset,a.candidate_run,a.candidate_prepared,a.candidate_pack,a.out)
if __name__=='__main__':main()
