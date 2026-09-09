#!/usr/bin/env python3
"""Read-only, policy-driven once monitor; it never mutates experiment artifacts."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,re
from pathlib import Path
UTC=dt.timezone.utc
def now():return dt.datetime.now(UTC)
def stamp(t):return t.isoformat().replace('+00:00','Z')
def parse(x):return dt.datetime.fromisoformat(x.replace('Z','+00:00'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def atomic(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def tail(p):return p.read_text(errors='replace')[-30000:] if p.is_file() else ''
def curve_last(path):
 if not path.is_file():return None
 rows=[]
 for line in path.read_text(errors='replace').splitlines():
  try: rows.append(json.loads(line))
  except json.JSONDecodeError: pass
 return rows[-1] if rows else None
def inspect(root,j):
 name=j.get('name',j.get('id'))
 if not isinstance(name,str) or not name:raise RuntimeError('monitor job needs id or name')
 run=root/j.get('run_dir',f"{j['stage']}/{j['dataset']}/{j['profile']}/{j.get('arm','D_JOINT')}/s42");log=root/j.get('log',f"logs/{name}.log");text=tail(log);kind=j.get('dataset')
 x=curve_last(run/'training_curve.jsonl') if kind=='h1' else None
 if x is None and kind=='h1':
  rows=[]
  for line in text.splitlines():
   try:
    q=json.loads(line)
    if isinstance(q,dict) and 'epoch' in q and 'source_val_ema' in q: rows.append(q)
   except json.JSONDecodeError: pass
  x=rows[-1] if rows else None
 if kind=='m1':
  if x is None:
   fs=sorted(run.glob('ema_epoch_*.json'));x=read(fs[-1]) if fs else None
  heartbeat=read(run/'heartbeat.json') if (run/'heartbeat.json').is_file() else None
 else: heartbeat=None
 done=(run/'train_receipt.json').is_file(); epoch=None;steps=None;elapsed=None;val=None
 if x:
  epoch=x.get('epoch');steps=x.get('step');elapsed=x.get('elapsed_seconds');val=x.get('source_val_ema',x.get('source_val'))
 if kind=='m1' and heartbeat:
  # EMA files alone define the completed epoch; heartbeat alone supplies live
  # step/elapsed status and must never be promoted to a completed epoch.
  steps=heartbeat.get('step',steps);elapsed=heartbeat.get('elapsed_seconds',elapsed)
 if done:
  r=read(run/'train_receipt.json');epoch=(len(r.get('curve',[])) if kind=='h1' else r.get('epochs',epoch));steps=r.get('steps',steps);val=r.get('selected',{}).get('source_val_ema',r.get('selected',{}).get('source_val',val))
 # exact numeric special values only; do not match ordinary words containing inf.
 bad=r'(?<![A-Za-z])(?:nan|\+?inf(?:inity)?)(?![A-Za-z])'
 trace='Traceback (most recent call last)' in text;oom=bool(re.search(r'out of memory|cuda error.*memory',text,re.I));nonfinite=bool(re.search(bad,text,re.I));effective='FAILED' if trace or oom or nonfinite else ('COMPLETED' if done else ('TRAINING' if epoch is not None or heartbeat is not None else j.get('status','QUEUED')));return {'name':name,'policy_status':j.get('status'),'effective_status':effective,'run_dir':str(run),'log':str(log),'done':done,'last_completed_epoch':epoch,'steps':steps,'total_elapsed_seconds':elapsed,'source_validation':val,'traceback':trace,'oom':oom,'nonfinite':nonfinite}
def frozen(root,policy):
 out={}
 groups=policy.get('frozen_code',policy.get('frozen_code_paths',{}))
 if not groups: groups={k:v for k,v in policy.items() if k.startswith('frozen_') and k.endswith('_code')}
 for group,items in groups.items():
  if isinstance(items,dict): items=[items]
  # Actual policy form is {absolute_path: sha256}; retain the older
  # [{path,sha256}] form for backward-compatible policy parsing.
  if len(items)==1 and isinstance(items[0],dict) and 'path' not in items[0]:items=[{'path':p,'sha256':d} for p,d in items[0].items()]
  rows=[]
  for raw in items:
   expected=None
   if isinstance(raw,dict): p=Path(raw['path']);expected=raw.get('sha256')
   else: p=Path(raw)
   p=p if p.is_absolute() else root.parents[1]/p
   actual=sha(p) if p.is_file() else None;rows.append({'path':str(p),'expected_sha256':expected,'actual_sha256':actual,'match':actual is not None and (expected is None or actual==expected)})
  out[group]=rows
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--force',action='store_true');a=p.parse_args();root=a.root.resolve();pp=root/'monitor_policy.json';policy=read(pp);t=now();due=policy.get('next_monitor_utc')
 if not a.force and due is not None and t<parse(due):print(json.dumps({'status':'NOT_DUE','next_monitor_utc':due}));return
 jobs=[inspect(root,j) for j in policy.get('jobs',[])];fr=frozen(root,policy);complete=bool(jobs) and all(j['done'] for j in jobs);queues=[]
 for rel in policy.get('queue_paths',[]):
  q=Path(rel);q=q if q.is_absolute() else root/q;row=read(q) if q.is_file() else {'status':'MISSING','error':str(q)};queues.append({'path':str(q),'status':row.get('status'),'error':row.get('error')})
 queue_failed=any(q['status']=='FAILED' for q in queues);failed=any(j['effective_status']=='FAILED' for j in jobs) or queue_failed
 report={'schema':'carrier_last1_v2_monitor_once','status':'FAILED' if failed else ('SOURCE_TRAINING_COMPLETE' if complete else 'MONITORED'),'monitor_utc':stamp(t),'jobs':jobs,'queues':queues,'frozen_code':fr,'note':'source validation is observational only'}
 with (root/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(report,sort_keys=True)+'\n')
 for row,live in zip(policy.get('jobs',[]),jobs):row['status']=live['effective_status']
 policy.update(last_monitor_utc=stamp(t),next_monitor_utc=None if complete or failed else stamp(t+dt.timedelta(seconds=600)),status=report['status']);atomic(pp,policy)
 summary=[]
 for j in jobs:
  code_rows=[x for group in fr.values() for x in group];dataset_group=f"frozen_{j['name'].split('_')[1]}_code" if '_' in j['name'] else None;dataset_code=fr.get(dataset_group,[])
  summary.append({'id':j['name'],'status':j['effective_status'],'done':j['done'],'traceback':j['traceback'],'oom':j['oom'],'nonfinite':j['nonfinite'],'epoch':j['last_completed_epoch'],'steps':j['steps'],'elapsed':j['total_elapsed_seconds'],'code_match':bool(code_rows) and all(x['match'] for x in code_rows),'dataset_code_match':bool(dataset_code) and all(x['match'] for x in dataset_code)})
 print(json.dumps({'status':report['status'],'jobs':summary,'queue_status':queues},sort_keys=True))
if __name__=='__main__':main()
