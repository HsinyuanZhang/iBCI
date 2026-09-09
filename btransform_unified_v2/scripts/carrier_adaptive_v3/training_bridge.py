#!/usr/bin/env python3
"""Conservative source-only bridge to frozen carrier_last1_v2 trainers.

It never imports trainer modules in-process: their sibling imports and module
constants are process-global.  It emits explicit subprocess commands only for
an exact v2 horizon; adaptive/new horizons fail closed with a copy map.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;V2=HERE.parent/'carrier_last1_v2';RESULTS=V2.parents[1]/'results/carrier_adaptive_v3'
FIXED={'m1':24,'h1':32}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def die(s):raise RuntimeError(s)
def require_root(p):
 p=p.resolve()
 if RESULTS not in (p,*p.parents):die(f'output must be below {RESULTS}')
 return p
def curve(dataset,run):
 run=run.resolve();rows=[]
 if dataset=='m1':
  for p in sorted(run.glob('ema_epoch_*.json')):
   x=json.loads(p.read_text());rows.append({'epoch':x.get('epoch'),'step':x.get('step'),'source_validation':x.get('source_val'),'path':str(p),'sha256':sha(p)})
 else:
  rec=run/'train_receipt.json'
  if not rec.is_file():die(f'missing {rec}')
  x=json.loads(rec.read_text())
  for row in x.get('curve',[]):rows.append({'epoch':row.get('epoch'),'step':row.get('step'),'source_validation':row.get('source_val_ema'),'sampler_endpoint_keep_sha256':row.get('sampler_endpoint_keep_sha256')})
 if not rows:die('no source-only curve found')
 return rows
def plan(a):
 out=require_root(a.out);run_root=require_root(a.run_root);horizon=FIXED[a.dataset]
 if a.max_epochs!=horizon:die(f'v2 {a.dataset} horizon is scheduler-bound at {horizon}; adaptive horizon requires the v3 copy map in training_bridge_README.md')
 if a.method!='adaptive':die('only method=adaptive is supported by this bridge')
 profile=a.profile
 if profile not in ('state','muscle','encoding'):die('adaptive bridge profile must be an explicit v3 candidate label')
 run=run_root/a.stage/a.dataset/profile/'D_JOINT'/'s42';pack=a.pack.resolve()
 if not pack.is_file():die(f'missing source-only carrier pack: {pack}')
 trainer=V2/('m1_train.py' if a.dataset=='m1' else 'h1_train.py')
 if a.dataset=='m1':
  common=[a.python,'-u',str(trainer),'--dest',str(run),'--arm','D_JOINT','--device',a.device,'--temporal-stage',a.stage,'--carrier-pack',str(pack)]
  commands=[common+['--stage',phase] for phase in ('prepare','preflight','train')]
 else:
  prepared=run_root/a.stage/'h1'/'prepared'/profile/'D_JOINT';prep=V2/'h1_prepare.py';adapter=V2/'outer_h1_authority_adapter.py'
  commands=[[a.python,'-u',str(prep),'--dest',str(prepared),'--surface','source','--carrier-pack',str(pack)]]
  if a.stage=='outer':commands.append([a.python,'-u',str(adapter),'--prepared',str(prepared)])
  commands=commands+[[a.python,'-u',str(trainer),'--prepared',str(prepared),'--dest',str(run),'--arm','D_JOINT','--stage',phase,'--device',a.device,'--cpu-threads','1'] for phase in ('preflight','train')]
 payload={'schema':'carrier_adaptive_v3_training_bridge_v1','status':'PLANNED_SOURCE_ONLY','dataset':a.dataset,'stage':a.stage,'method':a.method,'candidate_profile':profile,'max_epochs':a.max_epochs,'scheduler_horizon':'frozen_v2_exact','run':str(run),'carrier_pack':str(pack),'carrier_pack_sha256':sha(pack),'environment':{'CARRIER_STAGE':a.stage,'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},'commands':[{'argv':c,'env':{'CARRIER_STAGE':a.stage,'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}} for c in commands],'target_io':False,'target_scoring':False,'selection':'source validation only; earliest maximum among saved EMA epochs','v2_trainer':str(trainer),'v2_trainer_sha256':sha(trainer)}
 if out.exists():raise FileExistsError(out)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('plan-source');q.add_argument('--dataset',choices=('m1','h1'),required=True);q.add_argument('--stage',choices=('inner','outer'),required=True);q.add_argument('--method',default='adaptive');q.add_argument('--profile',required=True);q.add_argument('--pack',type=Path,required=True);q.add_argument('--max-epochs',type=int,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--run-root',type=Path,required=True);q.add_argument('--device',default='cuda:0');q.add_argument('--python',default=sys.executable)
 q=sub.add_parser('inspect-source-curve');q.add_argument('--dataset',choices=('m1','h1'),required=True);q.add_argument('--run',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='plan-source':plan(a)
 else:
  out=require_root(a.out);rows=curve(a.dataset,a.run);best=max(rows,key=lambda r:((r['source_validation'] or {}).get('equal_date_mean',(r['source_validation'] or {}).get('equal_session_mean',float('-inf'))),-int(r['epoch'])))
  if out.exists():raise FileExistsError(out)
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'schema':'carrier_adaptive_v3_source_curve_inspection_v1','dataset':a.dataset,'run':str(a.run.resolve()),'source_curve':rows,'earliest_source_maximum':best},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
