#!/usr/bin/env python3
"""Create a descriptive paired original-S42 versus mean-rate-S42 HO3 curve artifact.

Reads only the three JSON receipts from each supplied formal run.  It never
opens predictions, checkpoints, arrays, NWB/data, models, GPU, or official-test material.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Mapping
from summarize import (EPOCHS,HO,METRIC,ORIGINAL_VARIANT,CANDIDATE_VARIANT,_atomic_json,_atomic_text,_difference,_need,_read_run,_same_contract,_same_targets_and_starts,_sha)
HERE=Path(__file__).resolve().parent
LABELS={'original-s42':'Original SVD4 (seed42)','candidate-s42':'SVD3 + mean rate (seed42)'}
COLORS={'original-s42':'#1f77b4','candidate-s42':'#d62728'}

def _csv(path:Path,runs:list[Mapping[str,Any]])->None:
 fields=('run_id','label','epoch',METRIC,*[f'{x}_channel_variance_weighted_r2' for x in HO])
 tmp=path.with_suffix('.tmp')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in runs:
   for e in EPOCHS:
    row=r['curve'][str(e)];w.writerow({'run_id':r['id'],'label':LABELS[r['id']],'epoch':e,METRIC:row[METRIC],**{f'{x}_channel_variance_weighted_r2':row['per_session'][x]['channel_variance_weighted_r2'] for x in HO}})
 tmp.replace(path)

def _plot(png:Path,svg:Path,runs:list[Mapping[str,Any]])->None:
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 panels=(('HO3 mean',None),*((x,x) for x in HO));fig,axes=plt.subplots(2,2,figsize=(12.5,8.4),constrained_layout=True)
 fig.suptitle('Paired seed42 curves — visible HO3 calibration only',fontsize=15)
 for ax,(title,session) in zip(axes.flat,panels):
  for r in runs:
   ys=[r['curve'][str(e)][METRIC] if session is None else r['curve'][str(e)]['per_session'][session]['channel_variance_weighted_r2'] for e in EPOCHS]
   ax.plot(EPOCHS,ys,label=LABELS[r['id']],color=COLORS[r['id']],linewidth=2.1)
   e=r['selected']['epoch'];ax.scatter(e,ys[e-1],color=COLORS[r['id']],s=48,zorder=3)
  ax.set(title=title,xlabel='Epoch',ylabel='Channel variance-weighted R²',xticks=(1,3,6,12,18,24));ax.grid(alpha=.25);ax.legend(fontsize=9,frameon=False)
 fig.savefig(png,dpi=150);fig.savefig(svg);plt.close(fig)

def _endpoint(r:Mapping[str,Any], key:str)->dict[str,Any]: return r[key]
def _readme(a:Mapping[str,Any],b:Mapping[str,Any])->str:
 return f'''# Paired seed42 candidate curve

Descriptive comparison of one formal original-SVD4 seed42 run and one formal SVD3-plus-mean-rate seed42 run. Both curves cover all 24 EMA epochs on visible HO3 calibration (3,881 windows). This artifact contains no seed aggregate, statistical-significance claim, improvement claim, or official metric.

Selected markers are each run's own earliest maximum of the equal-session channel-variance-weighted HO3 metric. `curves.csv` holds every epoch and session value.

| Run | Selected epoch | Selected HO3 mean | Epoch-24 HO3 mean |
| --- | ---: | ---: | ---: |
| {LABELS[a['id']]} | {a['selected']['epoch']} | {a['selected'][METRIC]:.12g} | {a['curve']['24'][METRIC]:.12g} |
| {LABELS[b['id']]} | {b['selected']['epoch']} | {b['selected'][METRIC]:.12g} | {b['curve']['24'][METRIC]:.12g} |
'''

def run(args:argparse.Namespace)->dict[str,Any]:
 out=args.dest.resolve();_need(not out.exists(),f'--dest must be fresh: {out}')
 original=_read_run(args.original_s42,run_id='original-s42',seed=42,variant=ORIGINAL_VARIANT,old_schema=True)
 candidate=_read_run(args.candidate_s42,run_id='candidate-s42',seed=42,variant=CANDIDATE_VARIANT,old_schema=False)
 runs=[original,candidate]
 source=_same_contract(runs,'source_noncarrier_contract','source/sampler noncarrier contract');ho=_same_contract(runs,'ho_noncarrier_contract','HO noncarrier contract');targets=_same_targets_and_starts(runs)
 _need(original['b3s']==candidate['b3s'],'B3S initialization source mismatch')
 _need(original['carrier_variant']!=candidate['carrier_variant'],'candidate must be the only variant difference')
 deltas={str(e):_difference(candidate['curve'][str(e)],original['curve'][str(e)]) for e in EPOCHS}
 selected_delta=_difference(candidate['selected'],original['selected']);fixed24=_difference(candidate['curve']['24'],original['curve']['24'])
 body={'schema':'m1_muscle_r100_candidate_seed42_pair_plot_v1','status':'COMPLETED_PAIRED_DESCRIPTIVE_ONLY','scope':'One paired seed42 original/candidate comparison; no aggregate, significance, claim of improvement, or official metric.','input_policy':{'read_scope':'JSON receipts only; no predictions, checkpoints, data, NWB, NPZ, model, GPU, or official-test material','formal_epochs':24,'updates':159960,'full_ho3_windows':3881,'metric':METRIC},'code_sha256':{'plot_candidate_seed42_pair.py':_sha(Path(__file__).resolve()),'summarize.py':_sha(HERE/'summarize.py')},'input_receipts_sha256':{r['id']:r['input_receipts_sha256'] for r in runs},'invariants':{'source_and_sampler_noncarrier_contract':source,'ho_noncarrier_contract':ho,'ho_targets_starts_window_counts':targets,'same_b3s_initialization_source':True,'candidate_only_variant_difference':True},'runs':{r['id']:{k:v for k,v in r.items() if k not in {'source_noncarrier_contract','ho_noncarrier_contract'}} for r in runs},'selected_epoch_delta_candidate_minus_original':candidate['selected']['epoch']-original['selected']['epoch'],'selected_endpoint_delta_candidate_minus_original':selected_delta,'epoch24_delta_candidate_minus_original':fixed24,'all_24_same_epoch_delta_candidate_minus_original':deltas,'per_session_selected_endpoints':{r['id']:r['selected']['per_session'] for r in runs},'per_session_epoch24':{r['id']:r['curve']['24']['per_session'] for r in runs},'selection_interpretation':'Each selected epoch is its own earliest maximum equal-session metric on visible HO3 calibration.'}
 out.mkdir(parents=True);_atomic_json(out/'pair_summary.json',body);_csv(out/'curves.csv',runs);_plot(out/'curves.png',out/'curves.svg',runs);_atomic_text(out/'README.md',_readme(original,candidate))
 return {'status':body['status'],'summary':str(out/'pair_summary.json'),'csv':str(out/'curves.csv'),'png':str(out/'curves.png'),'svg':str(out/'curves.svg'),'readme':str(out/'README.md')}

def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original-s42',type=Path,required=True);p.add_argument('--candidate-s42',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))
if __name__=='__main__':main()
