#!/usr/bin/env python3
"""Outer source-only B/old-D/candidate-D queue; contains no target command."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
from protocol import SEED, atomic_json, protocol, sha
HERE=Path(__file__).resolve().parent

def die(x): raise RuntimeError(x)
def read(p):
 if not p.is_file(): die(f'missing {p}')
 x=json.loads(p.read_text())
 if not isinstance(x,dict):die(f'object required {p}')
 return x
def run(cmd,env,log):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a',encoding='utf8') as f:
  f.write('$ '+' '.join(cmd)+'\n');f.flush();subprocess.run(cmd,check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
def code_paths_ok(bindings,label):
 if not isinstance(bindings,dict) or not bindings:die(f'missing code binding {label}')
 for name,digest in bindings.items():
  p=Path(name)
  if not isinstance(digest,str) or not p.is_file() or sha(p)!=digest:die(f'code binding drift {label}: {name}')
def code_ok(row,label):code_paths_ok(row.get('source_code_sha256'),label)
def roster(ds):
 from source_seal import roster as sealed_roster
 return sealed_roster('outer',ds)
def pack(path,ds,profile):
 # source_seal owns the finite/shape/roster check.  The source-pack sidecar
 # additionally binds the bytes and, for M1, the frozen source fit.
 from source_seal import pack_info
 src,tgt=roster(ds); info=pack_info(path,ds,'outer',profile,src,tgt);meta=info['metadata'];side=read(path.with_suffix('.json'))
 if side!={**meta,'npz_sha256':info['sha256']}:die(f'carrier pack sidecar/plan metadata drift {path}')
 if ds=='m1':
  frozen=Path(meta.get('frozen_fit',''));digest=meta.get('frozen_fit_sha256')
  if (not frozen.is_file() or not isinstance(digest,str) or sha(frozen)!=digest
      or side.get('frozen_fit')!=meta.get('frozen_fit') or side.get('frozen_fit_sha256')!=digest):die(f'M1 frozen-fit binding drift {path}')
 return info
def m1_phase(row,path,arm,status,need_code=False):
 src,tgt=roster('m1')
 if (row.get('schema')!='m1_carrier_last1_v2' or row.get('status')!=status or row.get('arm')!=arm
     or row.get('seed')!=SEED or row.get('sources')!=src or row.get('targets')!=tgt):die(f'M1 phase identity drift {path}')
 if status=='PREPARED' and row.get('source_sessions')!=src:die(f'M1 prepare source roster drift {path}')
 if need_code:code_ok(row,str(path))
def m1_prepare_meta(rd,arm):m1_phase(read(rd/'prepare.json'),rd/'prepare.json',arm,'PREPARED')
def m1_preflight_meta(rd,arm):
 p=rd/'preflight.json';x=read(p);m1_phase(x,p,arm,'PASSED',need_code=True)
 if x.get('source_only') is not True or not isinstance(x.get('source_evidence'),dict):die(f'M1 preflight source evidence drift {p}')
def h1_prepared_meta(pre,arm,pi):
 src,tgt=roster('h1');manp=pre/'source'/'manifest.json';man=read(manp)
 if (man.get('schema')!='h1_carrier_last1_prepare_v1' or man.get('surface')!='source' or man.get('split_id')!='h1_outer_last1_19250120'
     or man.get('source_sessions')!=src or man.get('target_sessions')!=tgt or man.get('source_count')!=11
     or man.get('target_count')!=2 or sorted(man.get('records',{}))!=sorted(src)):die(f'H1 prepare identity/roster drift {manp}')
 auth=man.get('source_authority',{})
 if auth.get('source_sessions')!=src or auth.get('target_sessions')!=tgt or auth.get('target_records_opened')!=0:die(f'H1 prepare source authority drift {manp}')
 files=man.get('source_authority_files',{})
 for key,name in (('json_sha256','source_hc_plan.json'),('arrays_sha256','source_hc_plan_arrays.npz')):
  q=pre/'source'/name
  if not q.is_file() or files.get(key)!=sha(q):die(f'H1 prepare authority-file drift {q}')
 code_paths_ok(man.get('implementation_sha256'),str(manp))
 if pi is not None:
  records=man['records']
  if any(records[s].get('carrier_sha256')!=pi['array_sha256'][s] or records[s].get('validation',{}).get('carrier_sha256')!=pi['array_sha256'][s] for s in src):die(f'H1 prepared carrier-pack binding drift {manp}')
 clarification=pre/'outer_authority_clarification.json';x=read(clarification)
 if (x.get('schema')!='carrier_last1_v2_outer_h1_authority_erratum_v1' or x.get('status')!='SEALED' or x.get('stage')!='outer' or x.get('dataset')!='h1'
     or x.get('prepared_root')!=str(pre) or x.get('source_manifest')!=str(manp) or x.get('source_manifest_sha256')!=sha(manp)
     or x.get('source_sessions')!=src or x.get('target_sessions')!=tgt or x.get('historical_text_correction_only') is not True
     or x.get('algorithm_changed') is not False or x.get('source_arrays_rewritten') is not False):die(f'H1 clarification identity drift {clarification}')
 for key in ('source_hc_plan','source_hc_plan_arrays','adapter'):
  q=Path(x.get(key,''));digest=x.get(key+'_sha256')
  if not q.is_file() or not isinstance(digest,str) or sha(q)!=digest:die(f'H1 clarification SHA drift {clarification}: {key}')
 return manp
def h1_preflight_meta(rd,pre,arm,pi):
 manp=h1_prepared_meta(pre,arm,pi);p=rd/'preflight.json';x=read(p);src,tgt=roster('h1');split=x.get('split',x)
 if (x.get('schema')!='h1_carrier_last1_preflight_v1' or x.get('status')!='PASSED' or x.get('arm')!=arm or x.get('source_only') is not True
     or x.get('target_records_opened')!=0 or x.get('source_manifest_sha256')!=sha(manp)
     or split.get('source_sessions')!=src or split.get('target_sessions')!=tgt):die(f'H1 preflight arm/manifest binding drift {p}')
def phase_meta(ds,rd,profile,arm,pi):
 if ds=='m1':
  m1_prepare_meta(rd,arm);m1_preflight_meta(rd,arm);x=read(rd/'train_receipt.json');m1_phase(x,rd/'train_receipt.json',arm,'COMPLETED',need_code=True)
 else:
  pre=rd.parents[2]/'prepared'/profile/arm;h1_preflight_meta(rd,pre,arm,pi)
  rec,meta=read(rd/'train_receipt.json'),read(rd/'run_meta.json')
  if meta.get('schema')!='h1_carrier_last1_train_v1' or meta.get('arm')!=arm or meta.get('seed')!=SEED:die(f'H1 meta drift {rd}')
  if rec.get('schema')!='h1_carrier_last1_train_receipt_v1' or rec.get('meta',{}).get('arm')!=arm or rec.get('meta',{}).get('seed')!=SEED:die(f'H1 receipt drift {rd}')
  src,tgt=roster('h1')
  for x in (meta,rec.get('meta',{})):
   split=x.get('split',x)
   if split.get('source_sessions')!=src or split.get('target_sessions')!=tgt:die(f'H1 roster drift {rd}')
  code_ok(meta,str(rd/'run_meta.json'))
def complete(ds,rd,profile,arm,pi):
 """Pure source_seal verifier, also used before existing phase/seal reuse."""
 from source_seal import m1,h1
 phase_meta(ds,rd,profile,arm,pi)
 return m1(rd,'outer',arm,pi) if ds=='m1' else h1(rd,'outer',profile,arm,pi)
def cache_copy(old,new):
 """Copy literal old M1 aux cache into candidate profile cache byte-for-byte."""
 oldcache=old.parent.parent/'source_prepare_cache'; newcache=new.parent.parent/'source_prepare_cache'
 files=sorted(oldcache.glob('rsyn3_*.npz'))
 if not files:die(f'old literal auxiliary cache absent: {oldcache}')
 rows=[]
 newcache.mkdir(parents=True,exist_ok=True)
 for src in files:
  for source in (src,src.with_suffix('.json')):
   if not source.is_file():die(f'old cache companion absent: {source}')
   dst=newcache/source.name
   if dst.exists():
    if sha(dst)!=sha(source):die(f'candidate cache byte drift: {dst}')
   else:shutil.copyfile(source,dst)
   rows.append({'source':str(source.resolve()),'destination':str(dst.resolve()),'sha256':sha(source)})
 receipt=newcache/'old_rsyn3_cache_copy_receipt.json'; body={'schema':'carrier_last1_v2_m1_old_aux_cache_copy_v1','files':rows}
 if receipt.exists() and read(receipt)!=body:die(f'cache copy receipt drift: {receipt}')
 if not receipt.exists():atomic_json(receipt,body)
def valid_existing_seal(ds,rd,profile,arm,pi):
 expected=complete(ds,rd,profile,arm,pi); x=read(rd/'source_seal.json')
 for k,v in {'schema':'carrier_last1_v2_source_seal','status':'SEALED','dataset':ds,'stage':'outer','profile_id':profile,'arm':arm}.items():
  if x.get(k)!=v:die(f'existing seal identity drift {rd}')
 if x.get('receipt_sha256')!=expected['receipt_sha256'] or x.get('full_checkpoint_sha256')!=expected['full_checkpoint_sha256']:die(f'existing seal completed-run binding drift {rd}')
 if x.get('carrier_pack')!=pi:die(f'existing seal pack binding drift {rd}')
def valid_existing_gate(gate,dest,ds,profile):
 g=read(gate)
 if any(g.get(k)!=v for k,v in {'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'source','stage':'outer','dataset':ds,'profiles':['old',profile]}.items()):die(f'existing gate identity drift {gate}')
 expected={'B':('old','B_ACTIVITY_ONLY'),'old_D':('old','D_JOINT'),f'{profile}_D':(profile,'D_JOINT')};seals=g.get('source_seals')
 if not isinstance(seals,dict) or set(seals)!=set(expected):die(f'existing gate seal roster drift {gate}')
 for key,(p,a) in expected.items():
  path=dest/'outer'/ds/p/a/'s42'/'source_seal.json';binding=seals[key]
  if not path.is_file() or binding.get('path')!=str(path) or binding.get('sha256')!=sha(path):die(f'existing gate seal SHA/path drift {gate}: {key}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('m1','h1'),required=True);p.add_argument('--profile',choices=('encoding','state','muscle'),required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--dest',type=Path,required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--execute',action='store_true');a=p.parse_args();a.dest=a.dest.resolve()
 if a.dataset=='h1' and a.profile=='muscle':die('H1 has no muscle profile')
 packpath=a.dest/'packs'/'outer'/a.dataset/f'{a.profile}.npz'; jobs=[('old','B_ACTIVITY_ONLY',None),('old','D_JOINT',None),(a.profile,'D_JOINT',packpath)]
 def commands(profile,arm,cp):
  rd=a.dest/'outer'/a.dataset/profile/arm/'s42'; extra=[] if cp is None else ['--carrier-pack',str(cp)]; z=[]
  launch=[a.python,'-u']
  if cp:z.append([*launch,str(HERE/'carrier_prepare.py'),'--dataset',a.dataset,'--profile-id',profile,'--stage','outer','--surface','source','--dest',str(cp)])
  if a.dataset=='m1':
   z += [[*launch,str(HERE/'m1_train.py'),'--dest',str(rd),'--arm',arm,'--stage',phase,'--device',a.device,'--temporal-stage','outer',*extra] for phase in ('prepare','preflight','train')]
  else:
   pre=a.dest/'outer'/'h1'/'prepared'/profile/arm
   z += [[*launch,str(HERE/'h1_prepare.py'),'--dest',str(pre),'--surface','source',*extra],[*launch,str(HERE/'outer_h1_authority_adapter.py'),'--prepared',str(pre)]]
   z += [[*launch,str(HERE/'h1_train.py'),'--prepared',str(pre),'--dest',str(rd),'--arm',arm,'--stage',phase,'--device',a.device,'--cpu-threads','1'] for phase in ('preflight','train')]
  z.append([*launch,str(HERE/'source_seal.py'),'--run',str(rd),'--dataset',a.dataset,'--stage','outer','--profile-id',profile,'--arm',arm,*extra]); return z
 audit_command=[a.python,'-u',str(HERE/'paired_audit.py'),'--run-root',str(a.dest),'--dataset',a.dataset,'--stage','outer','--profiles','old',a.profile,'--dest',str(a.dest/'outer'/a.dataset/'source_gate.json'),'--phase','source']
 plan={'schema':'carrier_last1_v2_outer_source_queue','status':'PLANNED','stage':'outer','dataset':a.dataset,'candidate_profile':a.profile,'seed':SEED,'target_io':False,'target_preparation':False,'target_scoring':False,'protocol':protocol(),'jobs':[{'profile':q,'arm':r,'run':str(a.dest/'outer'/a.dataset/q/r/'s42'),'commands':commands(q,r,c)} for q,r,c in jobs],'paired_source_audit_command':audit_command}
 queue=a.dest/'queue_outer_source'/f'{a.dataset}_{a.profile}.json'
 if not a.execute:atomic_json(queue,plan);print(json.dumps(plan,indent=2));return
 if queue.exists() and read(queue).get('status')=='FAILED':die(f'previous FAILED queue is retained: {queue}')
 env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CARRIER_STAGE='outer'); state={**plan,'status':'RUNNING','completed':[]};atomic_json(queue,state)
 try:
  if packpath.exists(): pi=pack(packpath,a.dataset,a.profile)
  else:
   run([a.python,'-u',str(HERE/'carrier_prepare.py'),'--dataset',a.dataset,'--profile-id',a.profile,'--stage','outer','--surface','source','--dest',str(packpath)],env,a.dest/'logs'/f'outer_{a.dataset}_{a.profile}_pack.log');pi=pack(packpath,a.dataset,a.profile)
  done=[]
  for profile,arm,cp in jobs:
   rd=a.dest/'outer'/a.dataset/profile/arm/'s42'; receipt=rd/'train_receipt.json'; usepi=None if cp is None else pi
   log=a.dest/'logs'/f'outer_{a.dataset}_{profile}_{arm}.log'
   if receipt.exists():complete(a.dataset,rd,profile,arm,usepi)
   else:
    if any(rd.glob('resume_latest.pt')) or any(rd.glob('ema_epoch_*.pt')):die(f'partial fit refuses resume: {rd}')
    extra=[] if cp is None else ['--carrier-pack',str(cp)]
    if a.dataset=='m1':
     if profile==a.profile:cache_copy(a.dest/'outer'/'m1'/'old'/'D_JOINT'/'s42',rd)
     if not (rd/'prepare.json').exists():run([a.python,'-u',str(HERE/'m1_train.py'),'--dest',str(rd),'--arm',arm,'--stage','prepare','--device',a.device,'--temporal-stage','outer',*extra],env,log)
     m1_prepare_meta(rd,arm)
     if not (rd/'preflight.json').exists():run([a.python,'-u',str(HERE/'m1_train.py'),'--dest',str(rd),'--arm',arm,'--stage','preflight','--device',a.device,'--temporal-stage','outer',*extra],env,log)
     m1_preflight_meta(rd,arm)
     run([a.python,'-u',str(HERE/'m1_train.py'),'--dest',str(rd),'--arm',arm,'--stage','train','--device',a.device,'--temporal-stage','outer',*extra],env,log)
    else:
     pre=a.dest/'outer'/'h1'/'prepared'/profile/arm
     if not (pre/'source'/'manifest.json').is_file():run([a.python,'-u',str(HERE/'h1_prepare.py'),'--dest',str(pre),'--surface','source',*extra],env,log)
     # Reuse is allowed only after the 11-source/1925-01-20 authority and
     # implementation SHA bindings have been checked.
     if (pre/'outer_authority_clarification.json').is_file():h1_prepared_meta(pre,arm,usepi)
     if not (pre/'outer_authority_clarification.json').is_file():run([a.python,'-u',str(HERE/'outer_h1_authority_adapter.py'),'--prepared',str(pre)],env,log)
     h1_prepared_meta(pre,arm,usepi)
     if not (rd/'preflight.json').exists():run([a.python,'-u',str(HERE/'h1_train.py'),'--prepared',str(pre),'--dest',str(rd),'--arm',arm,'--stage','preflight','--device',a.device,'--cpu-threads','1'],env,log)
     h1_preflight_meta(rd,pre,arm,usepi)
     run([a.python,'-u',str(HERE/'h1_train.py'),'--prepared',str(pre),'--dest',str(rd),'--arm',arm,'--stage','train','--device',a.device,'--cpu-threads','1'],env,log)
    complete(a.dataset,rd,profile,arm,usepi)
   if (rd/'source_seal.json').exists():valid_existing_seal(a.dataset,rd,profile,arm,usepi)
   else:run([a.python,'-u',str(HERE/'source_seal.py'),'--run',str(rd),'--dataset',a.dataset,'--stage','outer','--profile-id',profile,'--arm',arm,*([] if cp is None else ['--carrier-pack',str(cp)])],env,log)
   done.append((profile,arm));state['completed'].append({'profile':profile,'arm':arm,'seal_sha256':sha(rd/'source_seal.json')});atomic_json(queue,state)
  gate=a.dest/'outer'/a.dataset/'source_gate.json'
  if gate.exists():
   valid_existing_gate(gate,a.dest,a.dataset,a.profile)
  else:run(audit_command,env,a.dest/'logs'/f'outer_{a.dataset}_{a.profile}_source_audit.log')
  state['status']='SOURCE_TRAINING_COMPLETE';state['source_gate_sha256']=sha(gate);atomic_json(queue,state)
 except Exception as e:
  state['status']='FAILED';state['error_type']=type(e).__name__;state['error']=str(e);atomic_json(queue,state);raise
if __name__=='__main__':main()
