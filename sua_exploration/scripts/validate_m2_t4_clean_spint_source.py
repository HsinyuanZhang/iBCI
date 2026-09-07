#!/usr/bin/env python3
"""Validate and receipt one held-in-only T4 source before its single replay."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL_SHA='c723e8f8e5ca37dd9c24d27eb117d7107bfd9d42a7de392e9224a1c21a9ec3cb'
ADDENDUM=ROOT/'sua_exploration/results/m2_t4_clean_spint_replication_v1/pre_heldout_addendum_v1.json'
ADDENDUM_SHA='f87d3d0fbd2231062c7c2b430bd8da7128d583dbf5a1010940b37bb9adb2866f'

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def need(c: bool, msg: str) -> None:
    if not c: raise ValueError(msg)

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,choices=(43,44),required=True); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--teacher-receipt',type=Path,required=True); ap.add_argument('--protocol',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    need(not a.out.exists(),'refusing overwrite')
    need(sha(a.protocol)==PROTOCOL_SHA,'protocol drift')
    need(ADDENDUM.is_file() and sha(ADDENDUM)==ADDENDUM_SHA,'missing or SHA-drifted pre-heldout addendum')
    addendum=json.loads(ADDENDUM.read_text())
    need(addendum.get('binds_protocol_receipt_sha256')==PROTOCOL_SHA and addendum.get('created_before_new_seed_heldout_access') is True,'invalid pre-heldout addendum')
    p=a.artifact.resolve(); required=['resolved_config.yaml','split_manifest.json','checkpoint_manifest.json','metrics_summary.csv','teacher_metadata.json']
    need(not [x for x in required if not (p/x).is_file()], 'incomplete source artifact')
    c=yaml.safe_load((p/'resolved_config.yaml').read_text()); d,m=c['data'],c['model']; split=json.loads((p/'split_manifest.json').read_text()); ck=json.loads((p/'checkpoint_manifest.json').read_text()); tm=json.loads((p/'teacher_metadata.json').read_text()); receipt=json.loads(a.teacher_receipt.read_text())
    expected={'seed':a.seed,'train':True,'test':True,'no_early_stopping':True,'require_baseline_validation':False}
    need(all(c.get(k)==v for k,v in expected.items()),'source run contract drift')
    need(c.get('ckpt_path') in (None,'','null'),'source must not replay checkpoint')
    need(c.get('trainer',{}).get('max_epochs')==12,'source epoch budget drift')
    need((d.get('task'),d.get('loso_fold'),d.get('calibration_n_trials'),d.get('random_calibration'),d.get('include_heldout_in_fit'),d.get('include_heldout_in_test'),d.get('side_feature_group'))==('m2',1,24,False,False,False,'t4'),'source data contract drift')
    need((m.get('variant'),m.get('require_clean_teacher_receipt'),m.get('freeze_decoder'),m.get('ssc_t4_prediction_consistency_weight'))==('B3S',True,True,0.0),'source model contract drift')
    need(split.get('heldout_evaluated_in_fit') is False and split.get('heldout_evaluated_in_test') is False,'source opened heldout')
    selected=receipt.get('selected_checkpoint',{}); need(tm.get('teacher_checkpoint_sha256')==selected.get('sha256'),'teacher receipt mismatch')
    cp=Path(str(ck.get('artifact_checkpoint_path',''))); need(cp.is_file() and ck.get('artifact_checkpoint_sha256')==sha(cp),'bad checkpoint hash')
    need(ck.get('selected_by_metric')=='val_heldin/r2_mean','non-heldin checkpoint selector')
    heldin = None
    for row in csv.DictReader((p/'metrics_summary.csv').open()):
        if row.get('split') == 'test_heldin': heldin = float(row['R2_variance_weighted']); break
    need(heldin is not None and math.isfinite(heldin),'missing or non-finite heldin test export')
    norm=split.get('native_t4_normalization',{}); need(norm.get('feature_group')=='t4' and norm.get('sha256') and isinstance(norm.get('train_sessions'),list),'missing T4 normalization')
    teacher={'checkpoint_sha256':selected['sha256'],'receipt_sha256':sha(a.teacher_receipt),'receipt_path':str(a.teacher_receipt.resolve())}
    checkpoint={'path':str(cp.resolve()),'sha256':sha(cp),'selected_by_metric':'val_heldin/r2_mean','max_epochs':12,'no_early_stopping':True}
    out={'schema_version':1,'seed':a.seed,'protocol':{'path':str(a.protocol.resolve()),'sha256':PROTOCOL_SHA},'pre_heldout_addendum':{'path':str(ADDENDUM.resolve()),'sha256':ADDENDUM_SHA},'source_artifact':str(p),'teacher_receipt':{'path':str(a.teacher_receipt.resolve()),'sha256':sha(a.teacher_receipt),'teacher_checkpoint_sha256':selected['sha256']},'teacher':teacher,'checkpoint':checkpoint,'arms':{'ordinary_t4':{'teacher':teacher,'checkpoint':checkpoint}},'t4_normalization':{'sha256':norm['sha256'],'train_sessions':norm['train_sessions']},'heldin_test_metric_present_and_finite':True,'heldin_performance_not_used_as_gate':True,'heldout_opened':False,'contract_runtime_noncatastrophic':True,'catastrophic_stop':False,'catastrophic_stop_semantics':'not_a_performance_threshold_contract_runtime_only','next_action':'one_frozen_local_heldout_replay_required'}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(a.out)
if __name__=='__main__': main()
