#!/usr/bin/env python3
"""Receipt-only Stage2 RT supervision-accounting closure (CPU, no model/GPU)."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
S0=ROOT/'sua_exploration/results/rt_simple_label_v1/stage0b/RT_SPARSE_ENDPOINT_STAGE0B_RECEIPT_v1.json'
S1=ROOT/'sua_exploration/results/rt_simple_label_v1/stage1/RT_SPARSE_ENDPOINT_STAGE1_RECEIPT_v1.json'
P2=ROOT/'sua_exploration/results/rt_simple_label_v1/stage2_preflight/RT_SPARSE_ENDPOINT_PRODUCTION_CPU_PARITY_v2.json'
OUT=ROOT/'sua_exploration/results/rt_simple_label_v1/stage2_preflight/RT_SPARSE_ENDPOINT_STAGE2_SUPERVISION_ACCOUNTING_v1.json'
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main()->None:
 if OUT.exists():raise FileExistsError(OUT)
 s0,s1,p2=(json.loads(p.read_text()) for p in (S0,S1,P2))
 if s0.get('status')!='PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU' or s1.get('status')!='PASS_STAGE1_SPARSE_ENDPOINT_AC4_CONSTRUCTIBLE_NO_GPU' or p2.get('status')!='PASS_PRODUCTION_T4D_STAGE1_PARITY_AND_NWB_PROVENANCE_NO_GPU':raise ValueError('sealed receipt status drift')
 names=sorted(s0['sessions']);
 if names!=sorted(s1['sessions']) or names!=sorted(p2['sessions']) or len(names)!=15:raise ValueError('15-session allowlist drift')
 nwbs=[]
 for n in names:
  row=s0['sessions'][n]['nwb'];p=Path(row['path'])
  if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError(f'NWB provenance drift: {n}')
  stage1_nwb=s1['sessions'][n]['nwb']
  if any(stage1_nwb.get(k)!=row.get(k) for k in ('path','bytes','sha256')):raise ValueError(f'Stage1 NWB provenance drift: {n}')
  nwbs.append({'session':n,**row})
 sem=sum(s0['sessions'][n]['m24']['endpoint_scalar_accounting']['raw_scalar_coordinates'] for n in names)
 dense_rows=sum(s0['sessions'][n]['m24']['endpoint_scalar_accounting']['dense_rt_retained_rows'] for n in names)
 dense=sum(s0['sessions'][n]['m24']['endpoint_scalar_accounting']['dense_rt_target_scalars'] for n in names)
 reach=sum(s1['sessions'][n]['reach_counts']['support'] for n in names)
 io=sum(p2['sessions'][n]['unique_endpoint_coordinate_scalars'] for n in names)
 if (reach,dense_rows,dense,sem,io)!=(1103,7855,15710,2764,5502):raise ValueError('frozen M24 accounting drift')
 payload={'schema':'rt_sparse_endpoint_stage2_supervision_accounting_v1','status':'PASS_RECEIPT_ONLY_SUPERVISION_ACCOUNTING_NO_GPU','bound_inputs':{str(p.name):{'path':str(p),'sha256':sha(p)} for p in (S0,S1,P2)},'implementation':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve())},'nwb_allowlist':nwbs,'scope':{'sessions':15,'support':'chronological M24 only','model_constructed':False,'gpu_context_created':False},'accounting':{'estimator_consumed_labels':{'reach_direction_scalars':reach,'dense_afc4_velocity_rows_2d':dense_rows,'dense_afc4_velocity_scalar_coordinates':dense,'dense_vs_reach_row_ratio':dense_rows/reach,'dense_vs_reach_scalar_ratio':dense/reach},'semantic_unique_endpoint_coordinate_payload':{'scalars':sem,'dense_scalar_coordinates':dense,'dense_vs_semantic_ratio':dense/sem},'production_bracket_interpolation_raw_coordinate_io':{'scalars':io,'dense_scalar_coordinates':dense,'dense_vs_actual_io_ratio':dense/io}},'interpretation':'Counts are distinct label/payload/IO accounting definitions only; they are not human annotation cost, effective N, or compute.'}
 OUT.parent.mkdir(parents=True,exist_ok=True);d=Path(tempfile.mkdtemp(prefix='.supervision-',dir=OUT.parent));t=d/OUT.name
 try:t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.chmod(t,0o444);os.replace(t,OUT);d.rmdir()
 except Exception:
  for x in d.iterdir():x.unlink()
  d.rmdir();raise
 print(OUT)
if __name__=='__main__':main()
