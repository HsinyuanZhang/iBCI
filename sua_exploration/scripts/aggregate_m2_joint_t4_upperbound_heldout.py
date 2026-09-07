#!/usr/bin/env python3
"""Fail-closed seed42 aggregate for the visible local M2 joint upper-bound replay."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'streaming_calibration_exp/outputs/streaming_calibration'
ARMS={'zero4':'zero4','t4':'t4','ts4':'ts4'}; S={'ses-2020-10-30-Run1','ses-2020-10-30-Run2','ses-2020-11-18-Run1','ses-2020-11-19-Run1','ses-2020-11-24-Run1','ses-2020-11-24-Run2'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def one(pat):
 x=sorted(p for p in OUT.glob(pat) if p.is_dir())
 if len(x)!=1:raise ValueError(f'expected one {pat}, got {x}')
 return x[0]
def read(a,src):
 p=one(f'm2_joint_t4_upperbound_v2_{a}_heldout_f1_s42_*');c=yaml.safe_load((p/'resolved_config.yaml').read_text());d,m=c['data'],c['model'];ck=json.loads((p/'checkpoint_manifest.json').read_text());sp=json.loads((p/'split_manifest.json').read_text())
 if (c.get('train'),c.get('test'),d.get('side_feature_group'),d.get('include_heldout_in_fit'),d.get('include_heldout_in_test'),d.get('query_start_trial'),m.get('freeze_decoder'))!=(False,True,ARMS[a],False,True,24,False):raise ValueError(f'{a}: heldout config contract')
 if ck.get('artifact_checkpoint_sha256')!=src['arms'][a]['checkpoint']['sha256']:raise ValueError(f'{a}: frozen final checkpoint drift')
 q=sp.get('heldout_query_window_audit',{})
 if set(q)!=S or any(x.get('support_trials')!=24 or x.get('query_start_trial')!=24 or x.get('window_size')!=50 or not x.get('full_window_disjoint') or x.get('eligible_windows',0)<=0 for x in q.values()):raise ValueError(f'{a}: full-window query contract')
 rows=[r for r in csv.DictReader((p/'metrics_per_session.csv').open()) if r.get('split')=='test_heldout'];sc={r['session']:float(r['R2_variance_weighted']) for r in rows}
 if set(sc)!=S or len(sc)!=6 or not all(math.isfinite(x) for x in sc.values()):raise ValueError(f'{a}: six finite heldout scores required')
 return {'artifact':str(p.resolve()),'checkpoint_sha256':ck['artifact_checkpoint_sha256'],'scores':sc,'query_window_audit':q}
def delta(x,y):
 d={s:x[s]-y[s] for s in sorted(S)}
 return {'per_session':d,'mean_r2':sum(d.values())/len(d),'positive_sessions':sum(v>0 for v in d.values()),'all_nonpositive':all(v<=0 for v in d.values())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-receipt',type=Path,required=True);ap.add_argument('--protocol-go',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 src=json.loads(a.source_receipt.read_text());go=json.loads(a.protocol_go.read_text())
 if src.get('screen')!='m2_joint_t4_upperbound_v2' or src.get('source_heldout_opened') is not False:raise ValueError('source receipt contract')
 if go.get('allow_local_heldout_replay') is not True or go.get('source_receipt_sha256')!=sha(a.source_receipt):raise ValueError('GO/source binding')
 arms={k:read(k,src) for k in ARMS};t4zero=delta(arms['t4']['scores'],arms['zero4']['scores']);t4ts=delta(arms['t4']['scores'],arms['ts4']['scores'])
 action='stop_seed42_no_go' if t4zero['mean_r2']<=0 and t4ts['mean_r2']<=0 else 'independent_review_required_before_any_seed43_44'
 z={'schema_version':1,'screen':'m2_joint_t4_upperbound_v2','scope':'visible local held-out-calib development replay; not EvalAI, hidden or formal','seed':42,'source_receipt':{'path':str(a.source_receipt.resolve()),'sha256':sha(a.source_receipt)},'protocol_go':{'path':str(a.protocol_go.resolve()),'sha256':sha(a.protocol_go)},'no_backward_optimizer_or_checkpoint_selection_on_heldout':True,'arms':arms,'joint_T4_minus_joint_zero4':t4zero,'joint_T4_minus_joint_TS4':t4ts,'seed42_action':action,'claim_limit':'seed42 is kill/no-go only; no success/effectiveness claim and no seeds43/44 without independent review'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(z,indent=2,sort_keys=True)+'\n');print(a.out)
if __name__=='__main__':main()
