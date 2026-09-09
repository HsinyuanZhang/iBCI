#!/usr/bin/env python3
"""Source-gated carrier-last1 runner; target preparation occurs only after paired seals/audit."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from protocol import PROFILES,OUT,protocol,atomic_json
HERE=Path(__file__).resolve().parent
def run(c,e): subprocess.run(c,check=True,env=e)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('inner','outer'),required=True);p.add_argument('--datasets',nargs='+',choices=('m1','h1'),default=('m1','h1'));p.add_argument('--profiles',nargs='+',choices=('encoding','state','muscle'),default=('encoding','state'));p.add_argument('--dest',type=Path,default=OUT);p.add_argument('--python',default=sys.executable);p.add_argument('--device',default='cuda:0');p.add_argument('--execute',action='store_true');a=p.parse_args();dest=a.dest.resolve(); profiles=('old',*a.profiles)
 if len(set(a.profiles))!=len(a.profiles):raise ValueError('duplicate profiles')
 if not a.execute: atomic_json(dest/'launch_plan.json',{'stage':a.stage,'datasets':a.datasets,'profiles':profiles,'protocol':protocol(),'gate':'all source trains -> B/old-D/new-D seals -> paired audit -> target preparation/score'});return
 env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CARRIER_STAGE=a.stage)
 run([a.python,str(HERE/'protocol.py')],env); jobs=[]
 for ds in a.datasets:
  for profile in profiles:
   if ds=='h1' and profile=='muscle': continue
   pack=dest/'packs'/a.stage/ds/f'{profile}.npz'
   if not (ds=='h1' and profile=='old'):run([a.python,str(HERE/'carrier_prepare.py'),'--dataset',ds,'--profile-id',profile,'--stage',a.stage,'--dest',str(pack)],env)
   # B is one shared baseline, housed under old; every D remains profile-distinct.
   arms=('B_ACTIVITY_ONLY','D_JOINT') if profile=='old' else ('D_JOINT',)
   for arm in arms:
    rd=dest/a.stage/ds/profile/arm/'s42';rd.mkdir(parents=True,exist_ok=True);extra=[] if arm=='B_ACTIVITY_ONLY' or (ds=='h1' and profile=='old') else ['--carrier-pack',str(pack)]
    if ds=='m1':
     for phase in ('prepare','preflight','train'):run([a.python,'-u',str(HERE/'m1_train.py'),'--dest',str(rd),'--arm',arm,'--stage',phase,'--device',a.device,'--temporal-stage',a.stage,*extra],env)
    else:
     prepared=dest/a.stage/ds/'prepared'/profile/arm;run([a.python,'-u',str(HERE/'h1_prepare.py'),'--dest',str(prepared),'--surface','source',*extra],env)
     for phase in ('preflight','train'):run([a.python,'-u',str(HERE/'h1_train.py'),'--prepared',str(prepared),'--dest',str(rd),'--arm',arm,'--stage',phase,'--device',a.device],env)
    sealextra=[] if arm=='B_ACTIVITY_ONLY' or (ds=='h1' and profile=='old') else ['--carrier-pack',str(pack)]
    run([a.python,str(HERE/'source_seal.py'),'--run',str(rd),'--dataset',ds,'--stage',a.stage,'--profile-id',profile,'--arm',arm,*sealextra],env);jobs.append((ds,profile,arm,rd,pack,extra))
 for ds in a.datasets: run([a.python,str(HERE/'paired_audit.py'),'--run-root',str(dest),'--dataset',ds,'--stage',a.stage,'--profiles',*profiles,'--dest',str(dest/a.stage/ds/'source_gate.json')],env)
 for ds,profile,arm,rd,pack,extra in jobs:
  targetextra=extra
  if arm=='D_JOINT' and not (ds=='h1' and profile=='old'):
   targetpack=dest/'packs'/a.stage/ds/'target'/f'{profile}.npz'
   run([a.python,str(HERE/'carrier_prepare.py'),'--dataset',ds,'--profile-id',profile,'--stage',a.stage,'--surface','target','--source-pack',str(pack),'--dest',str(targetpack)],env)
   targetextra=['--carrier-pack',str(targetpack)]
  if ds=='m1':run([a.python,'-u',str(HERE/'m1_score.py'),'--dest',str(rd),'--arm',arm,'--device',a.device,'--temporal-stage',a.stage,'--source-carrier-pack',str(pack),'--source-gate',str(dest/a.stage/ds/'source_gate.json'),*targetextra],env)
  else:
   prepared=dest/a.stage/ds/'prepared'/profile/arm;run([a.python,'-u',str(HERE/'h1_prepare.py'),'--dest',str(prepared),'--surface','target',*targetextra],env);run([a.python,'-u',str(HERE/'h1_train.py'),'--prepared',str(prepared),'--dest',str(rd),'--arm',arm,'--stage','score','--device',a.device],env)
if __name__=='__main__':main()
