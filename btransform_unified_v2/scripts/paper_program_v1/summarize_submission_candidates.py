#!/usr/bin/env python3
"""Validate local receipt-bound EvalAI recommendation inventory; never submits."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)
def m1(root: Path) -> dict[str,Any]:
    run=root/'results/rift_v1/m1_r100_concat_s42_formal_v1'; score=run/'score_receipt.json'; train=run/'train_receipt.json'
    s=json.loads(score.read_text()); t=json.loads(train.read_text())
    require(s.get('status')=='COMPLETED','M1 score receipt not completed')
    require(t.get('status')=='COMPLETED','M1 train receipt not completed')
    epochs={str(x) for x in range(1,25)}
    require(set(s.get('ema_by_epoch',{}))==epochs,'M1 EMA epoch map is not complete 1..24')
    require(set(s.get('checkpoint_sha256_by_epoch',{}))==epochs,'M1 checkpoint map is not complete 1..24')
    e=int(s['selection']['epoch']); checkpoint=run/f'epoch_{e:03d}.pt'
    require(checkpoint.is_file(),f'M1 selected checkpoint missing: {checkpoint}')
    actual=sha(checkpoint); require(actual==s['checkpoint_sha256_by_epoch'][str(e)],'M1 selected checkpoint SHA mismatch')
    row=s['ema_by_epoch'][str(e)]
    require(row.get('partial') is False and set(row.get('per_session',{}))=={'20121004','20121017','20121024'},'M1 selected row incomplete')
    return {'dataset':'M1','status':'ELIGIBLE_CANDIDATE_FINAL_RECOMMENDATION_PENDING_COMPARISONS','validation':'PASSED_LOCAL_RECEIPT_AND_SELECTED_FILE','cell':s['cell'],'weight_view':'EMA','selected_epoch':e,'checkpoint_path':str(checkpoint),'checkpoint_sha256':actual,'selection_rule':s['selection']['rule'],'local_score':row['equal_session_mean'],'per_session':row['per_session'],'score_receipt_sha256':sha(score),'train_receipt_sha256':sha(train),'official_result':None,'remaining_evidence':['complete joint final receipt','predeclared joint comparison/selection contract','final priority comparison against joint']}
def m2(root: Path) -> dict[str,Any]:
    run=root/'results/rift_v1/m2_r50_concat_s42_ext6_pick_v1'; score=run/'score_receipt.json'; manifest=run/'manifest.json'; audit=run/'validation_audit.json'; package=run/'selected_ema.pt'
    s=json.loads(score.read_text()); m=json.loads(manifest.read_text()); a=json.loads(audit.read_text())
    require(a.get('status')=='PASSED','M2 concat full-curve validation failed')
    require(s.get('status')=='COMPLETED' and s.get('view')=='EMA','M2 score receipt incomplete')
    require(set(s.get('ema_by_epoch',{}))=={str(x) for x in range(1,25)},'M2 EMA epoch map is not complete 1..24')
    e=int(s['selection']['epoch']); row=s['ema_by_epoch'][str(e)]; checkpoint=Path(m['checkpoint_bytes'][str(e)]['path'])
    require(e==9 and checkpoint.is_file() and sha(checkpoint)==m['checkpoint_bytes'][str(e)]['sha256'],'M2 selected checkpoint binding failed')
    require(row.get('partial') is False and row.get('n_windows')==15403 and len(row.get('per_session',{}))==6,'M2 selected row incomplete')
    require(package.is_file() and sha(package)==s['selected_ema_state']['sha256'],'M2 selected EMA package mismatch')
    d_run=root/'results/rift_v1/m2_r50_joint_d_s42_ext6_pick_v1'; d_audit=d_run/'validation_audit.json'; d_score=d_run/'score_receipt.json'
    require(d_audit.is_file() and d_score.is_file(),'M2 D42 ext6 evidence missing')
    da=json.loads(d_audit.read_text()); ds=json.loads(d_score.read_text())
    require(da.get('status')=='PASSED' and ds.get('status')=='COMPLETED' and ds['selection']['epoch']==13,'M2 D42 ext6 evidence invalid')
    return {'dataset':'M2','status':'ELIGIBLE_CANDIDATE_FINAL_RECOMMENDATION_PENDING_REMAINING_MECHANISM_SEEDS','validation':'PASSED_FULL_EXT6_CURVE_AND_SELECTED_EMA','cell':m['cell'],'weight_view':'EMA','selected_epoch':e,'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'selection_rule':s['selection']['rule'],'local_score':row['equal_session_mean'],'per_session':row['per_session'],'score_receipt_sha256':sha(score),'manifest_sha256':sha(manifest),'validation_audit_sha256':sha(audit),'selected_ema_sha256':sha(package),'joint_d42_ext6':{'status':'COMPLETED_SINGLE_SEED_NOT_PROMOTING','epoch':13,'local_score':ds['selection']['equal_session_mean'],'score_receipt_sha256':sha(d_score),'validation_audit_sha256':sha(d_audit),'delta_vs_concat_e9':ds['selection']['equal_session_mean']-row['equal_session_mean']},'official_result':None,'remaining_evidence':['complete remaining B/D mechanism seed receipts','three-seed paired ext4 mechanism summary','predeclared concat-versus-joint comparison/selection contract','final priority decision and any authorized packing receipt']}
def h1(root: Path) -> dict[str,Any]:
    p=root/'results/rift_v1/h1_r300_official_receipt_20260907/official_result.json'; o=json.loads(p.read_text())
    return {'dataset':'H1','status':'REFERENCE_FROZEN','validation':'PASSED_LOCAL_OFFICIAL_RECEIPT','submission_id':582073,'official_receipt_sha256':sha(p),'official':o,'recommendation':'retain frozen official reference; do not score-chase, repack, or recommend resubmission'}
def main() -> None:
    a=argparse.ArgumentParser(); a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); a.add_argument('--output',type=Path,required=True); ns=a.parse_args(); root=ns.root
    rows=[m1(root),m2(root),h1(root),{'dataset':'688','status':'EXTERNAL_PENDING_INTERFACE','validation':'NOT_APPLICABLE_EXTERNAL_OWNER','required':['strict manifest','weight/selection evidence','local score receipt','official receipt when available']}]
    out={'schema':'evalai_submission_candidate_summary_v2','submission_policy':'Read-only local validation; never posts, submits, or accesses EvalAI.','recommendations':rows}
    ns.output.parent.mkdir(parents=True,exist_ok=True); ns.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
