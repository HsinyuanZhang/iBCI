#!/usr/bin/env python3
"""Render the sealed outer carrier-last1 audit JSON into small report files."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):
 if not p.is_file():raise RuntimeError(f'missing required input: {p}')
 x=json.loads(p.read_text())
 if not isinstance(x,dict):raise RuntimeError(f'object required: {p}')
 return x
def require(x,msg):
 if not x:raise RuntimeError(msg)
def metric_rows(dataset,replay,chosen):
 arms=('B','old_D',f'{chosen}_D');require(set(replay)==set(arms),'final audit arm roster drift')
 fixed='target_epoch24_ema' if dataset=='m1' else 'fixed_e32_ema';selected='target_selected_ema' if dataset=='m1' else 'selected_ema';out=[]
 if dataset=='m1':
  for arm in arms:
   rows=replay[arm];require(isinstance(rows,list) and len(rows)==1,'M1 expected one target replay')
   got={r.get('kind'):r for r in rows[0].get('replay',[])};require(set(got)=={fixed,selected},'M1 checkpoint replay drift')
   for endpoint,key in (('fixed',fixed),('selected',selected)):
    r=got[key];require(isinstance(r.get('r2'),(int,float)) and isinstance(r.get('channel_variance_weighted_r2'),(int,float)),'M1 required metrics missing')
    out += [{'endpoint':endpoint,'arm':arm,'session':rows[0]['target'],'metric':'legacy_flattened_R2','value':float(r['r2'])},{'endpoint':endpoint,'arm':arm,'session':rows[0]['target'],'metric':'channel_variance_weighted_R2','value':float(r['channel_variance_weighted_r2'])}]
 else:
  for arm in arms:
   got={r.get('kind'):r for r in replay[arm].get('replay',[])};require(set(got)=={fixed,selected},'H1 checkpoint replay drift')
   for endpoint,key in (('fixed',fixed),('selected',selected)):
    r=got[key];a=r.get('per_session_r2',{});b=r.get('per_session_channel_variance_weighted_r2',{});require(set(a)==set(b) and a,'H1 per-session metrics missing')
    for session in sorted(a):out += [{'endpoint':endpoint,'arm':arm,'session':session,'metric':'legacy_flattened_R2','value':float(a[session])},{'endpoint':endpoint,'arm':arm,'session':session,'metric':'channel_variance_weighted_R2','value':float(b[session])}]
    out += [{'endpoint':endpoint,'arm':arm,'session':'__equal_session_mean__','metric':'legacy_flattened_R2','value':sum(map(float,a.values()))/len(a)},{'endpoint':endpoint,'arm':arm,'session':'__equal_session_mean__','metric':'channel_variance_weighted_R2','value':float(r['sessionmean_channel_variance_weighted_r2'])}]
 return out
def add_deltas(rows,chosen):
 for endpoint in ('fixed','selected'):
  for metric in ('legacy_flattened_R2','channel_variance_weighted_R2'):
   sessions=sorted({r['session'] for r in rows if r['endpoint']==endpoint and r['metric']==metric})
   for session in sessions:
    base={r['arm']:r['value'] for r in rows if r['endpoint']==endpoint and r['metric']==metric and r['session']==session}
    for arm,value in ((f'{chosen}_minus_B',base[f'{chosen}_D']-base['B']),('oldD_minus_B',base['old_D']-base['B']),(f'{chosen}_minus_oldD',base[f'{chosen}_D']-base['old_D'])):
     rows.append({'endpoint':endpoint,'arm':arm,'session':session,'metric':metric,'value':value})
def one(root,dataset,dest):
 base=root/'outer'/dataset;selp,gatep,finalp=(base/'profile_selection.json',base/'source_gate.json',base/'final_audit.json');sel,gate,final=read(selp),read(gatep),read(finalp)
 require(sel.get('schema')=='carrier_last1_v2_outer_profile_selection' and sel.get('status')=='PROMOTED_AND_LOCKED' and sel.get('dataset')==dataset,'profile selection invalid');chosen=sel.get('profile_id');require(isinstance(chosen,str) and chosen,'chosen profile missing')
 require(all(gate.get(k)==v for k,v in {'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'source','stage':'outer','dataset':dataset,'profiles':['old',chosen]}.items()),'source gate invalid')
 require(all(final.get(k)==v for k,v in {'schema':'carrier_last1_v2_paired_audit','status':'PASSED','phase':'final','stage':'outer','dataset':dataset,'profiles':['old',chosen]}.items()),'final audit invalid')
 require(final.get('source_gate_sha256')==sha(gatep),'final/source-gate SHA drift');require(sel.get('fixed_epochs')==(24 if dataset=='m1' else 32) and sel.get('seed')==42,'fixed epoch/seed drift');require(isinstance(sel.get('source_dates'),list) and isinstance(sel.get('target_dates'),list),'selection dates missing');pack=root/'packs'/'outer'/dataset/f'{chosen}.npz';require(pack.is_file() and sel.get('source_pack_sha256')==sha(pack),'selection source-pack SHA drift');seals={'selection_rule_sha256':root/'inner_selection_rule.preregister.json','metric_definition_erratum_sha256':root/'metric_definition_erratum.json','profile_projection_code_seal_sha256':root/'profile_projection_code_seal.json'}
 for key,path in seals.items():require(path.is_file() and sel.get(key)==sha(path),f'selection seal SHA drift: {key}')
 inner=Path(sel.get('inner_audit_path',''));require(inner.is_file() and sel.get('inner_audit_sha256')==sha(inner),'selection inner-audit SHA drift')
 rows=metric_rows(dataset,final.get('target_replay',{}),chosen);require(all(math.isfinite(r['value']) for r in rows),'nonfinite report metric');
 if dataset=='h1':
  for endpoint in ('fixed','selected'):
   for arm in ('B','old_D',f'{chosen}_D'):
    for metric in ('legacy_flattened_R2','channel_variance_weighted_R2'):
     vals=[r['value'] for r in rows if r['endpoint']==endpoint and r['arm']==arm and r['metric']==metric and r['session']!='__equal_session_mean__'];mean=[r['value'] for r in rows if r['endpoint']==endpoint and r['arm']==arm and r['metric']==metric and r['session']=='__equal_session_mean__'];require(len(mean)==1 and math.isclose(mean[0],sum(vals)/len(vals),abs_tol=1e-12),'H1 aggregate drift')
 add_deltas(rows,chosen)
 return {'dataset':dataset,'chosen_profile':chosen,'fixed_epochs':sel['fixed_epochs'],'selection_rule_sha256':sel.get('selection_rule_sha256'),'source_profile_sha256':sel.get('source_pack_sha256'),'source_dates':sel['source_dates'],'target_dates':sel['target_dates'],'input_sha256':{str(p):sha(p) for p in (selp,gatep,finalp)},'rows':rows}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=('m1','h1'));p.add_argument('--dest',type=Path);a=p.parse_args();root=a.root.resolve();datasets=(a.dataset,) if a.dataset else ('m1','h1');dest=(a.dest.resolve() if a.dest else root/'report')
 if dest.exists():raise FileExistsError(dest)
 report=[one(root,d,dest) for d in datasets];dest.mkdir(parents=True)
 fields=('dataset','endpoint','metric','session','arm','value');fixed=[];selected=[]
 for block in report:
  for row in block['rows']:(fixed if row['endpoint']=='fixed' else selected).append({'dataset':block['dataset'],**row})
 for name,rows in (('fixed_endpoint_summary.csv',fixed),('selected_endpoint_summary.csv',selected)):
  with (dest/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 summary={'schema':'carrier_last1_v2_report_v1','status':'PASSED','metric_definitions':{'legacy_flattened_R2':'historical scorer compatibility: float64 flattened/global-mean R2','channel_variance_weighted_R2':'diagnostic per-channel-centered variance-weighted R2'},'reports':report,'artifacts':{n:str((dest/n).resolve()) for n in ('fixed_endpoint_summary.csv','selected_endpoint_summary.csv','RESULTS.md')}}
 lines=['# Outer carrier-last1 results','','仅报告 seed 42 的公开开发末日期评估；不构成跨 seed 或独立测试集结论。','', '指标分开报告：`legacy_flattened_R2` 为历史 scorer 兼容口径；`channel_variance_weighted_R2` 为逐通道居中诊断口径。','']
 def table(block,endpoint,per_session=False):
  source=' / '.join(block.get('source_dates',[]));target=' / '.join(block.get('target_dates',[]));lines.extend([f"### {endpoint} endpoint",'', '| Dataset | Source → target | Metric | B | old-D | new-D | new-D − B | old-D − B | new-D − old-D |', '|---|---|---|---:|---:|---:|---:|---:|---:|'])
  rs=[r for r in block['rows'] if r['endpoint']==endpoint and (per_session or r['session']=='__equal_session_mean__' or block['dataset']=='m1')]
  for metric in ('legacy_flattened_R2','channel_variance_weighted_R2'):
   sessions=sorted({r['session'] for r in rs if r['metric']==metric})
   for session in sessions:
    d={r['arm']:r['value'] for r in rs if r['metric']==metric and r['session']==session};new=f"{block['chosen_profile']}_D";nb=f"{block['chosen_profile']}_minus_B";no=f"{block['chosen_profile']}_minus_oldD";lines.append(f"| {block['dataset'].upper()}{(' '+session) if per_session else ''} | {source} → {target} | {metric} | {d['B']:.6f} | {d['old_D']:.6f} | {d[new]:.6f} | {d[nb]:.6f} | {d['oldD_minus_B']:.6f} | {d[no]:.6f} |")
  lines.append('')
 for block in report:
  lines += [f"## {block['dataset'].upper()} ({block['chosen_profile']})",'',f"固定 epoch：{block['fixed_epochs']}。",''];table(block,'fixed')
  if block['dataset']=='h1':lines += ['#### H1 fixed per-session detail',''];table(block,'fixed',True)
  lines += ['#### Selected checkpoint（辅助）',''];table(block,'selected')
 md=dest/'RESULTS.md';md.write_text('\n'.join(lines)+'\n')
 # Summary is last so it can bind every generated artifact without self-reference.
 summary['artifacts']={n:{'path':str((dest/n).resolve()),'sha256':sha(dest/n)} for n in ('fixed_endpoint_summary.csv','selected_endpoint_summary.csv','RESULTS.md')}
 (dest/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
