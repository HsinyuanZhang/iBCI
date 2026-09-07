#!/usr/bin/env python3
"""Receipt one already-executed matched ordinary-T4 local held-out replay."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
PROTO_SHA='c723e8f8e5ca37dd9c24d27eb117d7107bfd9d42a7de392e9224a1c21a9ec3cb'
ADD_SHA='f87d3d0fbd2231062c7c2b430bd8da7128d583dbf5a1010940b37bb9adb2866f'
EXPECTED={'ses-2020-10-30-Run1','ses-2020-10-30-Run2','ses-2020-11-18-Run1','ses-2020-11-19-Run1','ses-2020-11-24-Run1','ses-2020-11-24-Run2'}
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x:bool,m:str)->None:
 if not x:raise ValueError(m)
def nwbs():
 rows=[]
 for p in sorted((ROOT/'SPINT-main/data/000953').rglob('*held-out-calib*.nwb')):
  rows.append({'session':p.name.split('_')[1].split('.')[0],'path':str(p.resolve()),'size_bytes':p.stat().st_size,'sha256':sha(p)})
 need(len(rows)==6 and {x['session'] for x in rows}==EXPECTED,'frozen heldout file set drift');return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(43,44),required=True);ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--source-receipt',type=Path,required=True);ap.add_argument('--teacher-receipt',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--addendum',type=Path,required=True);a=ap.parse_args()
 p=a.artifact.resolve();out=p/'heldout_t4_clean_spint_replication_provenance.json';need(not out.exists(),'refusing provenance overwrite');need(sha(a.protocol)==PROTO_SHA and sha(a.addendum)==ADD_SHA,'protocol/addendum SHA drift')
 s=json.loads(a.source_receipt.read_text());need(s.get('seed')==a.seed and s.get('protocol',{}).get('sha256')==PROTO_SHA and s.get('pre_heldout_addendum',{}).get('sha256')==ADD_SHA,'source receipt binding drift');need(s.get('heldin_performance_not_used_as_gate') is True and s.get('contract_runtime_noncatastrophic') is True and s.get('next_action')=='one_frozen_local_heldout_replay_required','source receipt is not contract-only authorization')
 required=['resolved_config.yaml','split_manifest.json','checkpoint_manifest.json','teacher_metadata.json','metrics_per_session.csv'];need(not [x for x in required if not(p/x).is_file()],'incomplete heldout artifact')
 c=yaml.safe_load((p/'resolved_config.yaml').read_text());d,m=c['data'],c['model'];split=json.loads((p/'split_manifest.json').read_text());ck=json.loads((p/'checkpoint_manifest.json').read_text());tm=json.loads((p/'teacher_metadata.json').read_text());r=json.loads(a.teacher_receipt.read_text());src_ck=s['checkpoint']; src_teacher=s['teacher']
 need((c.get('seed'),c.get('train'),c.get('test'),c.get('optimized_metric'),c.get('require_baseline_validation'))==(a.seed,False,True,None,False),'test-only config drift')
 need((d.get('task'),d.get('loso_fold'),d.get('calibration_n_trials'),d.get('random_calibration'),d.get('include_heldout_in_fit'),d.get('include_heldout_in_test'),d.get('query_start_trial'),d.get('side_feature_group'))==('m2',1,24,False,False,True,24,'t4'),'heldout data contract drift')
 need((m.get('variant'),m.get('require_clean_teacher_receipt'),m.get('freeze_decoder'),m.get('ssc_t4_prediction_consistency_weight'))==('B3S',True,True,0.0),'heldout model drift')
 need(str(Path(str(c.get('ckpt_path'))).resolve())==src_ck['path'],'heldout checkpoint path drift');actual=Path(str(ck.get('artifact_checkpoint_path','')));need(actual.is_file() and sha(actual)==src_ck['sha256'] and ck.get('source_checkpoint_sha256')==src_ck['sha256'],'heldout checkpoint bytes drift')
 need(tm.get('teacher_checkpoint_sha256')==src_teacher['checkpoint_sha256']==r.get('selected_checkpoint',{}).get('sha256'),'teacher checkpoint mismatch');need(src_teacher['receipt_sha256']==sha(a.teacher_receipt) and s.get('teacher_receipt',{}).get('path')==str(a.teacher_receipt.resolve()),'teacher receipt mismatch')
 need(split.get('heldout_evaluated_in_fit') is False and split.get('heldout_evaluated_in_test') is True,'heldout split drift');audit=split.get('heldout_query_window_audit');need(isinstance(audit,dict) and set(audit)==EXPECTED,'query audit sessions drift')
 for session,row in audit.items():
  need(row.get('support_trials')==24 and row.get('query_start_trial')==24 and row.get('window_size')==50 and row.get('full_window_disjoint') is True,'query contract drift');need(row.get('query_trials',0)>0 and row.get('eligible_windows',0)>0,'empty query');need(row.get('minimum_window_start_padded_bin')==row.get('raw_query_start_bin',-50)+49,'support history leakage')
 norm=split.get('native_t4_normalization',{});need(norm.get('feature_group')=='t4' and norm.get('sha256')==s['t4_normalization']['sha256'],'T4 normalization drift')
 payload={'schema_version':1,'seed':a.seed,'protocol':{'path':str(a.protocol.resolve()),'sha256':PROTO_SHA},'pre_heldout_addendum':{'path':str(a.addendum.resolve()),'sha256':ADD_SHA},'source_receipt':{'path':str(a.source_receipt.resolve()),'sha256':sha(a.source_receipt)},'clean_teacher_receipt':{'path':str(a.teacher_receipt.resolve()),'sha256':sha(a.teacher_receipt),'selected_teacher_checkpoint_sha256':src_teacher['checkpoint_sha256']},'frozen_source_checkpoint':src_ck,'test_artifact_checkpoint':{'path':str(actual.resolve()),'sha256':sha(actual)},'six_heldout_calibration_nwbs':nwbs(),'support_contract':{'trial_range':[0,24],'chronological':True,'t4_target_labels':'24 support-trial target-direction labels only'},'query_contract':{'trial_range':[24,None],'window_size_bins':50,'full_history_disjoint':True,'per_session_audit':audit},'heldout_backward_optimizer_or_selection':False}
 out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(out)
if __name__=='__main__':main()
