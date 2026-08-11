#!/usr/bin/env python3
"""Receipt-first, resumable supervisor for the frozen RT Stage-2 45-cell matrix.

``--preflight`` and ``--print-only`` never launch a process.  ``--execute``
is intentionally explicit and sequential; it is the only mode that can use a
GPU via the established clean RT runner.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONTRACT=ROOT/'sua_exploration/docs/RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md'
READINESS=ROOT/'sua_exploration/results/rt_simple_label_v1/stage2_preflight/RT_SPARSE_ENDPOINT_STAGE2_ROOT_READINESS_REVIEW_v2.json'
RUNNER=ROOT/'streaming_calibration_exp/scripts/run_rt_clean_nested_loso.py'
SURFACES={"matrix_supervisor":Path(__file__).resolve(),"runner":RUNNER,"datamodule":ROOT/'streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py',"falcon_dataset":ROOT/'streaming_calibration_exp/src/data/falcon_datamodule.py',"t4d_loader":ROOT/'streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py',"outer_evaluator":ROOT/'streaming_calibration_exp/src/rt_clean_nested_loso_eval.py',"base_experiment":ROOT/'streaming_calibration_exp/configs/experiment/rt_clean_nested_loso_m24.yaml',"data_config":ROOT/'streaming_calibration_exp/configs/data/rt_nested_loso_m24.yaml'}
DATA=ROOT/'sua_exploration/data/dandi_000688/sub-C'; ARMS=('rt_sparse_endpoint_t4d','afc4_vel','zero4'); FOLDS=tuple(range(15)); SEED=42
TEACHER=ROOT/'SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt'

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def tree_digest(root:Path,pattern:str)->str:
 h=hashlib.sha256()
 for p in sorted(root.glob(pattern)):
  if p.is_file():h.update(str(p.relative_to(ROOT)).encode());h.update(sha(p).encode())
 return h.hexdigest()
def root_relative(path:Path)->str:
 return path.resolve().relative_to(ROOT.resolve()).as_posix()
def resolve_readiness_path(value:Any)->Path:
 raw=str(value)
 candidate=Path(raw)
 if candidate.is_absolute() or '..' in candidate.parts:raise ValueError('readiness paths must be ROOT-relative without parent traversal')
 resolved=(ROOT/candidate).resolve()
 if ROOT.resolve() not in resolved.parents and resolved!=ROOT.resolve():raise ValueError('readiness path escapes ROOT')
 return resolved
def launch_inputs()->dict[str,Any]:
 if not TEACHER.is_file():raise ValueError('teacher checkpoint missing')
 return {"configs_yaml_tree":{"path":str((ROOT/'streaming_calibration_exp/configs').resolve()),"sha256":tree_digest(ROOT/'streaming_calibration_exp/configs','**/*.yaml')},"src_py_tree":{"path":str((ROOT/'streaming_calibration_exp/src').resolve()),"sha256":tree_digest(ROOT/'streaming_calibration_exp/src','**/*.py')},"teacher_checkpoint":{"path":str(TEACHER.resolve()),"bytes":TEACHER.stat().st_size,"sha256":sha(TEACHER)}}
def atomic_json(path:Path,value:dict[str,Any])->None:
 if path.exists():raise FileExistsError(path)
 path.parent.mkdir(parents=True,exist_ok=True); d=Path(tempfile.mkdtemp(prefix='.matrix-',dir=path.parent)); t=d/path.name
 try:t.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.chmod(t,0o444);os.replace(t,path);d.rmdir()
 except Exception:
  for x in d.iterdir():x.unlink()
  d.rmdir();raise
def git_info()->dict[str,Any]:
 try:
  return {"commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"dirty":bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())}
 except Exception:return {"commit":None,"dirty":None}
def cells()->list[dict[str,Any]]:
 return [{"fold":f,"arm":a,"seed":SEED,"run_id":f"rt_stage2_{a}_f{f:02d}_s42","fresh_fit":True,"exactly_once_outer_eval":True,"output_key":f"f{f:02d}_{a}"} for f in FOLDS for a in ARMS]
def manifest()->dict[str,Any]:
 r=json.loads(READINESS.read_text());
 if r.get('status')!='PASS_ROOT_REVIEW_STAGE2_MATRIX_READY_NOT_LAUNCHED':raise ValueError('root readiness not launchable')
 bound=r.get('bound_files',{})
 for name,path in {"contract":CONTRACT,**SURFACES}.items():
  row=bound.get(name,{})
  if resolve_readiness_path(row.get('path',''))!=path.resolve() or str(row.get('path'))!=root_relative(path) or row.get('sha256')!=sha(path):raise ValueError(f'readiness bound-file drift: {name}')
 inputs=launch_inputs()
 for name,row in inputs.items():
  bound_row=bound.get(name,{})
  target=Path(row['path']).resolve()
  if resolve_readiness_path(bound_row.get('path',''))!=target or str(bound_row.get('path'))!=root_relative(target) or bound_row.get('sha256')!=row['sha256'] or (name=='teacher_checkpoint' and bound_row.get('bytes')!=row['bytes']):raise ValueError(f'readiness launch-input drift: {name}')
 nwbs=[]
 for p in sorted(DATA.glob('sub-C_ses-RT-*_behavior+ecephys.nwb')):nwbs.append({"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha(p)})
 if len(nwbs)!=15:raise ValueError('expected exact 15 RT NWBs')
 return {"schema":"rt_sparse_endpoint_stage2_matrix_manifest_v1","status":"PREPARED_NOT_LAUNCHED","contract":{"path":str(CONTRACT),"sha256":sha(CONTRACT)},"root_readiness":{"path":str(READINESS),"sha256":sha(READINESS)},"surfaces":{k:{"path":str(v),"sha256":sha(v)} for k,v in SURFACES.items()},"launch_inputs":inputs,"nwb_allowlist":nwbs,"environment":{"python":sys.version,"cuda_visible_devices":os.getenv('CUDA_VISIBLE_DEVICES'),"git":git_info()},"matrix":{"folds":list(FOLDS),"arms":list(ARMS),"seed":SEED,"cells":cells(),"fresh_fit_count":45,"outer_eval_count":45,"base_experiment":"rt_clean_nested_loso_m24","arm_override_only":True}}
def validate_cell(cell:dict[str,Any], receipt:dict[str,Any], manifest_sha256:str|None=None)->None:
 required=("selection_receipt_sha256","config_sha256","checkpoint_sha256","split_manifest_sha256")
 if any(not isinstance(receipt.get(k),str) or len(receipt[k])!=64 for k in required):raise ValueError('missing bound fit artifact SHA')
 if manifest_sha256 is not None and receipt.get('matrix_manifest_sha256')!=manifest_sha256:raise ValueError('matrix manifest SHA mismatch')
 if receipt.get('schema')!='rt_sparse_endpoint_stage2_cell_closure_v2':raise ValueError('cell closure schema mismatch')
 paths=receipt.get('artifact_paths')
 if not isinstance(paths,dict) or set(paths)!={'selection_receipt','config','checkpoint','split_manifest','outer_receipt'}:raise ValueError('complete artifact paths required')
 for key in paths:
  p=Path(paths[key]); digest=receipt.get(f'{key}_sha256' if key!='selection_receipt' else 'selection_receipt_sha256')
  if not p.is_file() or sha(p)!=digest:raise ValueError(f'artifact path/SHA drift: {key}')
 outer=receipt.get('outer_receipt');
 if not isinstance(outer,dict):raise ValueError('missing outer receipt')
 if (outer.get('arm'),outer.get('outer_loso_fold'),outer.get('seed'))!=(cell['arm'],cell['fold'],SEED):raise ValueError('outer arm/fold/seed mismatch')
 if outer.get('target_backpropagation') is not False or outer.get('model_state_three_point_unchanged') is not True:raise ValueError('outer no-BP/state proof failed')
 if not isinstance(outer.get('matched_query_window_identity'),dict):raise ValueError('outer query window identity absent')
 if receipt.get('outer_receipt_sha256') != sha(Path(paths['outer_receipt'])):raise ValueError('outer receipt hash mismatch')
 if json.loads(Path(paths['outer_receipt']).read_text())!=outer:raise ValueError('outer receipt closure payload mismatch')
 selection=json.loads(Path(paths['selection_receipt']).read_text()); split=json.loads(Path(paths['split_manifest']).read_text())
 if selection.get('status')!='PASS_FIT_INNER_SELECTION_ONLY' or selection.get('arm')!=cell['arm'] or selection.get('outer_loso_fold')!=cell['fold'] or selection.get('seed')!=SEED:raise ValueError('selection identity mismatch')
 if Path(selection.get('best_model_path','')).resolve()!=Path(paths['checkpoint']).resolve() or Path(selection.get('config_path','')).resolve()!=Path(paths['config']).resolve() or Path(selection.get('split_manifest_path','')).resolve()!=Path(paths['split_manifest']).resolve():raise ValueError('selection artifact chain mismatch')
 if (selection.get('best_model_sha256'),selection.get('config_sha256'),selection.get('split_manifest_sha256')) != (receipt['checkpoint_sha256'],receipt['config_sha256'],receipt['split_manifest_sha256']):raise ValueError('selection artifact SHA chain mismatch')
 if (outer.get('checkpoint_sha256'),outer.get('config_path'),outer.get('selection_receipt_path'),outer.get('fit_split_manifest')) != (receipt['checkpoint_sha256'],paths['config'],paths['selection_receipt'],paths['split_manifest']):raise ValueError('outer artifact chain mismatch')
 if split.get('requested_side_feature_group')!=cell['arm'] or split.get('outer_loso_fold')!=cell['fold'] or split.get('target_session')!=outer.get('outer_target_session'):raise ValueError('split/outer identity mismatch')
 q=outer['matched_query_window_identity']; session=outer.get('outer_target_session')
 if set(q)!={session}:raise ValueError('query audit must contain exactly the outer target session')
 audit=q[session]
 if not isinstance(audit,dict) or not all(isinstance(audit.get(k),str) and len(audit[k])==64 for k in ('ordered_window_start_sha256','ordered_target_covariate_evalmask_sha256','ordered_query_identity_sha256')):raise ValueError('strong query digest absent')

def audit_resume_state(man:dict[str,Any], root:Path, manifest_path:Path)->dict[str,Any]:
 """Audit already-closed cells without opening data or launching a worker.

 This small pure accounting boundary is shared by ``--execute`` and tests. It
 makes resume semantics explicit: every existing closure must validate against
 the immutable matrix manifest, while absent cells remain pending. A malformed
 closure fails closed instead of being silently retrained or overwritten.
 """
 manifest_sha256=sha(manifest_path)
 closed=[]; pending=[]
 expected_keys=set()
 for c in man['matrix']['cells']:
  key=str(c['output_key'])
  if key in expected_keys:raise ValueError(f'duplicate matrix output key: {key}')
  expected_keys.add(key)
  close=root/'cells'/f'{key}.json'
  if close.exists():
   validate_cell(c,json.loads(close.read_text()),manifest_sha256)
   closed.append(key)
  else:
   pending.append(key)
 return {
  'cells_total':len(man['matrix']['cells']),
  'closed_cells':len(closed),
  'pending_cells':len(pending),
  'closed_output_keys':closed,
  'pending_output_keys':pending,
  'all_closed':not pending,
 }
def summarize(values:dict[str,float])->dict[str,Any]:
 names=sorted(values);v=np.asarray([values[n] for n in names]);remove=max(range(len(names)),key=lambda i:(abs(v[i]),-i));keep=np.delete(v,remove)
 return {"ordered":[{"session":n,"delta":float(values[n])} for n in names],"mean":float(v.mean()),"median":float(np.median(v)),"positive":int((v>0).sum()),"zero":int((v==0).sum()),"negative":int((v<0).sum()),"leave_largest_absolute_out_mean":float(keep.mean()),"removed_session":names[remove],"mean_ge_003":bool(v.mean()>=.03),"median_ge_003":bool(np.median(v)>=.03),"primary_gate_pass":bool(v.mean()>0 and np.median(v)>0 and (v>0).sum()>len(v)/2)}
def aggregate(man:dict[str,Any],root:Path)->dict[str,Any]:
 records=[]
 for c in man['matrix']['cells']:
  p=root/'cells'/f"{c['output_key']}.json"
  if not p.is_file():raise ValueError(f'missing terminal cell {c["output_key"]}')
  x=json.loads(p.read_text());validate_cell(c,x,sha(root/'STAGE2_MATRIX_MANIFEST_v1.json'));records.append((c,x))
 by={(c['fold'],c['arm']):x for c,x in records}; deltas={}
 for fold in FOLDS:
  q=[by[(fold,a)]['outer_receipt']['matched_query_window_identity'] for a in ARMS]
  if not(q[0]==q[1]==q[2]):raise ValueError(f'fold {fold}: query-window identity mismatch')
  t,z,f=(by[(fold,a)]['outer_receipt']['r2_variance_weighted'] for a in ('rt_sparse_endpoint_t4d','zero4','afc4_vel')); session=by[(fold,'rt_sparse_endpoint_t4d')]['outer_receipt']['outer_target_session'];deltas[session]=(float(t-z),float(t-f))
 if len(deltas)!=15:raise ValueError('outer-session identity duplicate/missing')
 return {"schema":"rt_sparse_endpoint_stage2_matrix_aggregate_v1","status":"PASS_MATRIX_TERMINAL","manifest_sha256":sha(root/'STAGE2_MATRIX_MANIFEST_v1.json'),"cells":45,"t4d_minus_zero4":summarize({k:v[0] for k,v in deltas.items()}),"t4d_minus_full":summarize({k:v[1] for k,v in deltas.items()})}
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--print-only',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--accelerator',choices=('cpu','gpu'),default='gpu');p.add_argument('--device',default='cuda');a=p.parse_args()
 if sum((a.preflight,a.print_only,a.execute))!=1:raise ValueError('choose exactly one mode')
 m=manifest();mp=a.output_root/'STAGE2_MATRIX_MANIFEST_v1.json'
 if a.preflight:print(json.dumps({"status":"PASS_MATRIX_PREFLIGHT_NO_LAUNCH","cells":len(m['matrix']['cells']),"manifest":m},indent=2));return
 if a.print_only:
  for c in m['matrix']['cells']:print(' '.join(map(str,[sys.executable,RUNNER,'train','--fold',c['fold'],'--arm',c['arm'],'--seed',42,'--run-id',c['run_id'],'--accelerator',a.accelerator,'--devices',1,'--print-only'])))
  return
 if not mp.exists():atomic_json(mp,m)
 elif json.loads(mp.read_text())!=m:raise ValueError('existing manifest identity drift')
 resume=audit_resume_state(m,a.output_root,mp)
 for c in m['matrix']['cells']:
  close=a.output_root/'cells'/f"{c['output_key']}.json"
  if close.exists():validate_cell(c,json.loads(close.read_text()),sha(mp));continue
  if launch_inputs()!=m['launch_inputs'] or {k:{"path":str(v),"sha256":sha(v)} for k,v in SURFACES.items()}!=m['surfaces']:raise RuntimeError('launch inputs drifted after immutable manifest')
  run_root=ROOT/'streaming_calibration_exp/logs/train/runs'; found=list(run_root.glob(f"*rid-{c['run_id']}_f{c['fold']}_s42"))
  if len(found)>1:raise RuntimeError(f"ambiguous fit output for {c['output_key']}")
  if found:
   run=found[0]
   if not (run/'rt_nested_selection_receipt.json').is_file() or not (run/'split_manifest.json').is_file():raise RuntimeError(f"incomplete prior fit refuses resume: {c['output_key']}")
  else:
   train=[sys.executable,str(RUNNER),'train','--fold',str(c['fold']),'--arm',c['arm'],'--seed','42','--run-id',c['run_id'],'--accelerator',a.accelerator,'--devices','1']
   if subprocess.run(train,cwd=ROOT/'streaming_calibration_exp').returncode:raise RuntimeError(f"fresh fit failed: {c['output_key']}")
   found=list(run_root.glob(f"*rid-{c['run_id']}_f{c['fold']}_s42"))
   if len(found)!=1:raise RuntimeError(f"ambiguous/missing fresh fit output for {c['output_key']}")
   run=found[0]
  selection=run/'rt_nested_selection_receipt.json'; split=run/'split_manifest.json'
  if not selection.is_file() or not split.is_file():raise RuntimeError(f"missing fit receipts for {c['output_key']}")
  sel=json.loads(selection.read_text()); ckpt=Path(sel['best_model_path']); cfg=Path(sel['config_path']); outer=a.output_root/'outer'/f"{c['output_key']}.json"
  if not outer.exists():
   evaluate=[sys.executable,str(RUNNER),'eval','--config',str(cfg),'--checkpoint',str(ckpt),'--split-manifest',str(split),'--selection-receipt',str(selection),'--output',str(outer),'--outer-fold',str(c['fold']),'--device',a.device]
   if subprocess.run(evaluate,cwd=ROOT/'streaming_calibration_exp').returncode:raise RuntimeError(f"one-shot outer eval failed: {c['output_key']}")
  if not outer.is_file():raise RuntimeError(f"outer receipt missing: {c['output_key']}")
  o=json.loads(outer.read_text()); closure={"schema":"rt_sparse_endpoint_stage2_cell_closure_v2","matrix_manifest_sha256":sha(mp),"cell":c,"selection_receipt_sha256":sha(selection),"config_sha256":sha(cfg),"checkpoint_sha256":sha(ckpt),"split_manifest_sha256":sha(split),"outer_receipt_sha256":sha(outer),"artifact_paths":{"selection_receipt":str(selection.resolve()),"config":str(cfg.resolve()),"checkpoint":str(ckpt.resolve()),"split_manifest":str(split.resolve()),"outer_receipt":str(outer.resolve())},"outer_receipt":o}
  validate_cell(c,closure,sha(mp));atomic_json(close,closure)
 agg=aggregate(m,a.output_root);atomic_json(a.output_root/'STAGE2_MATRIX_AGGREGATE_v1.json',agg)
if __name__=='__main__':main()
