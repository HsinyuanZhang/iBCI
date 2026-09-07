#!/usr/bin/env python3
"""Fail-closed source gate: B3TS+T4 may be no worse than matched T4 by .03."""
import csv, hashlib, json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]; SCREEN="m2_m24_b3ts_t4_v1"; FEAS_SHA="6fbfea441acb0e9f98b75fae40b48bbf757f939eb4b4671cc10f69ddd7024171"
GROUP={"t4":("B3S","t4"),"b3ts":("B3TS","t4")}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def one(pat):
 h=list((ROOT/"streaming_calibration_exp/outputs/streaming_calibration").glob(pat));
 if len(h)!=1: raise ValueError(f"expected one {pat}, got {h}")
 return h[0]
def read(g):
 p=one(f"{SCREEN}_source_{g}_m2_f1_s42_*"); c=yaml.safe_load((p/'resolved_config.yaml').read_text()); d,m,t=c['data'],c['model'],c['trainer']; v,s=GROUP[g]
 want=("m2",1,42,v,s,24,False,False,False,0,12); got=(d.get('task'),d.get('loso_fold'),c.get('seed'),m.get('variant'),d.get('side_feature_group'),d.get('calibration_n_trials'),d.get('random_calibration'),d.get('include_heldout_in_fit'),d.get('include_heldout_in_test'),d.get('query_start_trial'),t.get('max_epochs'))
 if got!=want: raise ValueError(f"{g} source config mismatch {got}")
 split=json.loads((p/'split_manifest.json').read_text());
 if split.get('heldout_evaluated_in_fit') or split.get('heldout_evaluated_in_test'): raise ValueError('heldout source access')
 rows=list(csv.DictReader((p/'metrics_summary.csv').open())); x=[r for r in rows if r['split']=='test_heldin'];
 if len(x)!=1 or int(float(x[0]['M']))!=24: raise ValueError('source heldin metric invalid')
 ck=json.loads((p/'checkpoint_manifest.json').read_text()); q=Path(ck['artifact_checkpoint_path']);
 if not q.is_file() or sha(q)!=ck['artifact_checkpoint_sha256']: raise ValueError('checkpoint receipt')
 return {'path':str(p.resolve()),'score':float(x[0]['R2_variance_weighted']),'checkpoint':{'path':str(q.resolve()),'sha256':sha(q)}}
def main():
 feas=ROOT/f"sua_exploration/results/{SCREEN}/feasibility.json"; 
 if sha(feas)!=FEAS_SHA: raise ValueError('feasibility drift')
 arms={g:read(g) for g in GROUP}; delta=arms['b3ts']['score']-arms['t4']['score']; out=ROOT/f"sua_exploration/results/{SCREEN}/aggregate_source.json"
 if out.exists(): raise FileExistsError(out)
 out.write_text(json.dumps({'schema_version':1,'formal_heldout_evaluated':False,'feasibility_sha256':FEAS_SHA,'arms':arms,'b3ts_minus_t4':delta,'gate':{'threshold':-0.03,'pass':delta>=-0.03,'rule':'source gate only; does not claim effectiveness'}},indent=2)+'\n'); print(out)
if __name__=='__main__': main()
