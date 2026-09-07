#!/usr/bin/env python3
"""Strict three-seed T4 versus clean-SPINT matched local-replay aggregate."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

PROTO_SHA='c723e8f8e5ca37dd9c24d27eb117d7107bfd9d42a7de392e9224a1c21a9ec3cb'
ADD_SHA='f87d3d0fbd2231062c7c2b430bd8da7128d583dbf5a1010940b37bb9adb2866f'
S={'ses-2020-10-30-Run1','ses-2020-10-30-Run2','ses-2020-11-18-Run1','ses-2020-11-19-Run1','ses-2020-11-24-Run1','ses-2020-11-24-Run2'}
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,m):
 if not x:raise ValueError(m)
def scores_artifact(p:Path):
 x={}
 for r in csv.DictReader((p/'metrics_per_session.csv').open()):
  if r.get('split')=='test_heldout':x[r['session']]=float(r['R2_variance_weighted'])
 need(set(x)==S and all(math.isfinite(v) for v in x.values()),f'{p}: invalid six heldout scores');return x
def reference(p:Path):
 x=json.loads(p.read_text());q={k:float(v) for k,v in x.get('per_session_r2',{}).items()};need(set(q)==S and all(math.isfinite(v) for v in q.values()),f'{p}: invalid clean reference');need(math.isclose(float(x.get('mean_r2_equal_session')),sum(q.values())/6,rel_tol=0,abs_tol=1e-12),f'{p}: clean recorded mean drift');return x,q
def check_clean(c,prov,protocol_path,source_receipt,seed):
 need(c.get('role')=='clean_full_spint_matched_primary_baseline','clean reference role drift');need(c.get('protocol_receipt',{}).get('sha256')==PROTO_SHA,'clean reference protocol drift');need(c.get('source_gate',{}).get('path')==str(source_receipt.resolve()) and c.get('source_gate',{}).get('sha256')==sha(source_receipt),'clean source receipt drift')
 need(c.get('clean_teacher_receipt')==prov.get('clean_teacher_receipt'),'T4/clean teacher mismatch');d=c.get('descriptor_contract',{});need(all(d.get(k)==v for k,v in {'calibration_trials':24,'query_start_trial':24,'window_size_bins':50,'side_feature_group':'none','calibration_target_labels_used':False,'backward_gradients':False,'optimizer_updates':False,'checkpoint_selection':False}.items()),'clean no-label descriptor contract drift')
 need(c.get('six_heldout_calibration_nwbs')==prov.get('six_heldout_calibration_nwbs'),'clean NWC fingerprint drift');need(c.get('query_window_audit')==prov.get('query_contract',{}).get('per_session_audit'),'clean query audit drift')
 return reference_record_scores(c)
def reference_record_scores(c):
 q={k:float(v) for k,v in c.get('per_session_r2',{}).items()};need(set(q)==S and all(math.isfinite(v) for v in q.values()),'invalid clean six scores');need(math.isclose(float(c.get('mean_r2_equal_session')),sum(q.values())/6,rel_tol=0,abs_tol=1e-12),'clean recorded mean drift');return q
def check_new(seed:int,artifact:Path,clean:Path,source:Path):
 prov=json.loads((artifact/'heldout_t4_clean_spint_replication_provenance.json').read_text());src=json.loads(source.read_text());need(prov.get('seed')==seed and prov.get('protocol',{}).get('sha256')==PROTO_SHA and prov.get('pre_heldout_addendum',{}).get('sha256')==ADD_SHA,'new seed provenance binding drift');need(prov.get('source_receipt',{}).get('path')==str(source.resolve()) and prov.get('source_receipt',{}).get('sha256')==sha(source),'source receipt provenance drift');need(src.get('seed')==seed and src.get('protocol',{}).get('sha256')==PROTO_SHA and src.get('pre_heldout_addendum',{}).get('sha256')==ADD_SHA and src.get('heldin_performance_not_used_as_gate') is True and src.get('contract_runtime_noncatastrophic') is True and src.get('next_action')=='one_frozen_local_heldout_replay_required','source receipt/addendum drift');need(prov.get('heldout_backward_optimizer_or_selection') is False,'heldout update/selection drift');c=json.loads(clean.read_text());need(c.get('formal_heldout_evaluated') is False and c.get('hidden_evalai_evaluated') is False,'clean reference scope drift');return scores_artifact(artifact),check_clean(c,prov,None,source,seed),prov,c
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--addendum',type=Path,required=True);ap.add_argument('--seed42-aggregate',type=Path,required=True);ap.add_argument('--seed42-clean',type=Path,required=True)
 for s in (43,44):ap.add_argument(f'--seed{s}-artifact',type=Path,required=True);ap.add_argument(f'--seed{s}-clean',type=Path,required=True);ap.add_argument(f'--seed{s}-source-receipt',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();need(not a.out.exists(),'refusing aggregate overwrite');need(sha(a.protocol)==PROTO_SHA and sha(a.addendum)==ADD_SHA,'protocol/addendum SHA drift')
 old=json.loads(a.seed42_aggregate.read_text());need(sha(a.seed42_aggregate)=='0b53f53f4bccca8b760aedb633e6c0fbb3a654d6928377b6a36a21ceb37a6141','seed42 aggregate drift');need(sha(a.seed42_clean)=='6ca177a7aacef5150c1ef75ec89c54242f3b6a70189a39463dac491bf01d2d3a','seed42 clean reference drift')
 t42={k:float(v) for k,v in old['per_session_r2']['ordinary_t4'].items()};c42_record,c42=reference(a.seed42_clean);need(set(t42)==S,'seed42 T4 session drift')
 data={42:(t42,c42)}; receipts={42:{'aggregate_sha256':sha(a.seed42_aggregate),'clean_reference_sha256':sha(a.seed42_clean)}}
 for s in (43,44):
  t,c,prov,clean=check_new(s,getattr(a,f'seed{s}_artifact'),getattr(a,f'seed{s}_clean'),getattr(a,f'seed{s}_source_receipt'));data[s]=(t,c);receipts[s]={'source_receipt_sha256':sha(getattr(a,f'seed{s}_source_receipt')),'t4_provenance_sha256':sha(getattr(a,f'seed{s}_artifact')/'heldout_t4_clean_spint_replication_provenance.json'),'clean_reference_sha256':sha(getattr(a,f'seed{s}_clean')),'teacher_receipt_sha256':prov['clean_teacher_receipt']['sha256']}
 deltas={s:{k:data[s][0][k]-data[s][1][k] for k in sorted(S)} for s in (42,43,44)};seed_means={s:float(np.mean(list(deltas[s].values()))) for s in deltas};session_mean={k:float(np.mean([deltas[s][k] for s in deltas])) for k in sorted(S)};v=np.array(list(session_mean.values()));rng=np.random.default_rng(20260801);boot=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(10000)])
 out={'schema_version':1,'protocol':{'path':str(a.protocol.resolve()),'sha256':PROTO_SHA},'pre_heldout_addendum':{'path':str(a.addendum.resolve()),'sha256':ADD_SHA},'scope':'matched local heldout-calibration replay; not hidden EvalAI/final test','seed42_reused_not_rerun':True,'label_access_disclosure':{'ordinary_t4':'24 chronological support-trial target-direction labels','clean_full_spint':'M24 neural support only; no target labels'},'raw_scores_per_seed':{str(s):{'ordinary_t4':data[s][0],'clean_full_spint':data[s][1],'ordinary_t4_equal_session_mean':float(np.mean(list(data[s][0].values()))),'clean_full_spint_equal_session_mean':float(np.mean(list(data[s][1].values())))} for s in data},'provenance_hashes':receipts,'per_seed_session_deltas_t4_minus_clean_spint':deltas,'seed_mean_deltas':seed_means,'seed_averaged_session_deltas':session_mean,'primary_rule':'mean across three seed means >= +0.03, all 3 seed means positive, and >=5/6 seed-averaged session deltas positive','primary_effective':float(np.mean(list(seed_means.values())))>=.03 and all(v>0 for v in seed_means.values()) and sum(x>0 for x in session_mean.values())>=5,'primary_summary':{'three_seed_mean_delta':float(np.mean(list(seed_means.values()))),'positive_seed_means':sum(x>0 for x in seed_means.values()),'positive_seed_averaged_sessions':sum(x>0 for x in session_mean.values())},'descriptive_only':{'session_block_bootstrap_n':10000,'session_block_bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'exact_wilcoxon_two_sided_p':float(wilcoxon(v,alternative='two-sided',method='exact').pvalue)}}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(a.out)
if __name__=='__main__':main()
