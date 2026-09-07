#!/usr/bin/env python3
"""Fixed-auth, write-once r5 aggregate; no legacy auth CLI escape hatch."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from t4_m30_experiment_a_r5_authorization import require_claim, RECEIPT
from aggregate_t4_m30_experiment_a_v3 import summarize, decide
ARMS=('z4','ph4','ac4','mb4','b4','ls4'); REFS=('t4','ts4'); SEEDS=(42,43,44); EPOCHS=tuple(range(5,13)); SESSIONS=('sub-C_ses-CO-20151103','sub-C_ses-CO-20151104','sub-C_ses-CO-20151106','sub-C_ses-CO-20151109','sub-C_ses-CO-20151110','sub-C_ses-CO-20151112')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,m):
 if not x:raise ValueError(m)
def load(path:Path,arm:str,seed:int,*,new:bool,status_dir:Path|None,train_sha:str,ref_hashes:dict)->tuple[np.ndarray,dict]:
 need(path.is_file(),f'missing artifact {path}')
 if not new:need(sha(path)==ref_hashes[arm][str(seed)],f'unqualified reference {path}')
 d=json.loads(path.read_text());q=d.get('protocol',{});need((d.get('variant'),d.get('seed'),d.get('signal_view'),d.get('epoch_list'))==('B3S',seed,'sua',list(EPOCHS)),f'artifact identity {path}');need((q.get('calibration_n'),q.get('pool_size'),q.get('total_epochs'),q.get('burn_in_epochs'))==(30,30,12,4),f'protocol drift {path}');need(d.get('no_test_files_evaluated') is True and d.get('uses_backward_gradients') is False,f'test/gradient drift {path}')
 mpath=Path(d['run_metadata_path']);need(mpath.is_file() and sha(mpath)==d.get('run_metadata_sha256'),f'metadata binding {path}');m=json.loads(mpath.read_text());s=m.get('side_features',{});need((m.get('status'),m.get('variant'),m.get('seed'),m.get('task'),tuple(m.get('split_counts',[])),m.get('max_units_exclusive'),m.get('signal_view'),s.get('group'),s.get('side_dim'),s.get('pool_size'),s.get('feature_version'),m.get('held_out_test_evaluated'),m.get('teacher_sha256'),m.get('train_val_manifest_sha256'))==('completed','B3S',seed,'CO',(27,6,6),100,'sua',arm,4,30,1,False,'9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d','4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9'),f'metadata drift {path}');need(m.get('teacher_checkpoint')=='/home/xinyuan/Work_host/SPINT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt' and m.get('train_val_manifest')=='/home/xinyuan/Work_host/SPINT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json',f'metadata path drift {path}');expected_norm='a7ee643d066bd3db9ce6e2b178527bb6655d90c485fb5a0151e5babb8fe255ad' if arm=='ph4' else '293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0';need(s.get('normalization_sha256')==expected_norm,f'normalizer drift {path}');contract=s.get('descriptor_contract',{});need((arm!='ls4') or (contract.get('raw_refit_from_label_permuted_directions_before_normalization') is True and contract.get('aligned_intercept_b_copied_from_ordinary_t4') is True and contract.get('label_permutation_seed')==seed),f'LS4 permutation contract {path}');t=m.get('training',{});need((t.get('calibration_n_trials'),t.get('max_epochs'),t.get('no_early_stopping'),t.get('checkpoint_every_epoch'),t.get('loss_mode'),t.get('learning_rate'),t.get('batch_size'),t.get('freeze_decoder'),t.get('identity_mode'),t.get('deterministic'))==(30,12,True,True,'task_only',1e-4,32,False,'calibrated',True) and m.get('t4_logit_residual',{}).get('enabled') is False,f'training/residual drift {path}')
 evidence={'artifact_sha256':sha(path),'run_metadata_sha256':sha(mpath)}
 if new:
  sp=status_dir/f'{arm}_s{seed}.json';need(sp.is_file(),f'missing status {sp}');st=json.loads(sp.read_text());need((st.get('arm'),st.get('seed'),st.get('status'),st.get('exit_code'))==(arm,seed,'completed',0),f'bad status {sp}');need(st.get('result_sha256')==sha(path) and st.get('metadata_sha256')==sha(mpath),f'status artifact/metadata closure {sp}')
  cp=mpath.parent/'post_run_cost_receipt.json';need(cp.is_file(),f'missing cost {cp}');c=json.loads(cp.read_text());need(c.get('run_metadata_sha256')==sha(mpath) and c.get('train_variant_source_sha256')==train_sha and c.get('accelerator')=='gpu',f'cost provenance {cp}')
  for k in ('fit_wall_clock_seconds','cuda_peak_memory_allocated_bytes','cuda_peak_memory_reserved_bytes'):need(isinstance(c.get(k),(int,float)) and c[k]>0,f'bad cost {k}')
  need(st.get('cost_sha256')==sha(cp),f'status cost closure {sp}');evidence['operational_cost']={k:c[k] for k in ('fit_wall_clock_seconds','cuda_peak_memory_allocated_bytes','cuda_peak_memory_reserved_bytes')}
 rows=[]
 for e in EPOCHS:
  r=d['per_epoch'][str(e)]['per_session_r2'];need(sorted(r)==list(SESSIONS),f'session drift {path}');rows.append([r[x] for x in SESSIONS])
 x=np.asarray(rows,float);need(np.isfinite(x).all(),f'NaN result {path}');return x,evidence
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--status-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 need(not a.out.exists(),'aggregate output collision'); auth=require_claim();pre=json.loads(RECEIPT.read_text());refs=pre['qualified_reference_artifacts'];train_sha=pre['source_sha256']['sua_exploration/scripts/train_variant_dandi688.py'];data={};evidence={}
 for arm in ARMS+REFS:
  xs=[]
  for seed in SEEDS:
   x,ev=load((a.result_dir if arm in ARMS else a.reference_dir)/f'{arm}_s{seed}.json',arm,seed,new=arm in ARMS,status_dir=a.status_dir if arm in ARMS else None,train_sha=train_sha,ref_hashes=refs);xs.append(x);evidence[f'{arm}_s{seed}']=ev
  data[arm]=np.stack(xs)
 rng=np.random.default_rng(20260802);contrasts={}
 for arm in ARMS:
  for refname in ('z4','t4','ts4'):
   if arm==refname:continue
   st=summarize(data[arm]-data[refname],rng);a_seed=data[arm].mean((1,2));b_seed=data[refname].mean((1,2));u=float(np.sqrt(a_seed.var(ddof=1)/3+b_seed.var(ddof=1)/3));cor=float(np.corrcoef(a_seed,b_seed)[0,1]);st['paired_vs_unpaired']={'unpaired_seed_mean_se':u,'seed_correlation':cor if np.isfinite(cor) else None,'seed_correlation_undefined_reason':None if np.isfinite(cor) else 'zero variance across three seed means','paired_se_is_smaller':st['seed_mean_se_paired']<u};st['per_seed_dispersion']={'arm_seed_scores':a_seed.tolist(),'reference_seed_scores':b_seed.tolist(),'arm_sd':float(a_seed.std(ddof=1)),'reference_sd':float(b_seed.std(ddof=1))};st['decision']=decide(st);contrasts[f'{arm}_minus_{refname}']=st
 primary={}
 for left,right in (('t4','z4'),('t4','ls4'),('t4','mb4'),('t4','ts4')):
  st=summarize(data[left]-data[right],rng);a_seed=data[left].mean((1,2));b_seed=data[right].mean((1,2));u=float(np.sqrt(a_seed.var(ddof=1)/3+b_seed.var(ddof=1)/3));cor=float(np.corrcoef(a_seed,b_seed)[0,1]);st['paired_vs_unpaired']={'unpaired_seed_mean_se':u,'seed_correlation':cor if np.isfinite(cor) else None,'seed_correlation_undefined_reason':None if np.isfinite(cor) else 'zero variance across three seed means','paired_se_is_smaller':st['seed_mean_se_paired']<u};st['per_seed_dispersion']={'arm_seed_scores':a_seed.tolist(),'reference_seed_scores':b_seed.tolist(),'arm_sd':float(a_seed.std(ddof=1)),'reference_sd':float(b_seed.std(ddof=1))};st['decision']=decide(st);primary[f'{left}_minus_{right}']=st
 suff={}
 for arm in ('ph4','ac4','mb4','b4'):
  c,z=contrasts[f'{arm}_minus_t4'],contrasts[f'{arm}_minus_z4'];ok=c['paired_two_se_lower']>=-.03 and c['hierarchical_bootstrap']['lower_95']>=-.03 and z['decision']=='effective';suff[arm]={'sufficient':ok,'noninferiority_margin':-.03,'component_minus_t4_lower_paired_2se':c['paired_two_se_lower'],'component_minus_t4_lower_hierarchical_95':c['hierarchical_bootstrap']['lower_95'],'component_vs_z4_state':z['decision']}
 absolute={arm:{'mean_r2':float(x.mean()),'seed_mean_r2':x.mean((1,2)).tolist(),'session_mean_r2':x.mean((0,1)).tolist()} for arm,x in data.items()}
 out={'schema_version':5,'status':'completed','authorization_id':auth['authorization_id'],'nonce_claimed':True,'formal_test_used':False,'conditional_row_shuffle_launched':False,'arms':list(ARMS),'qualified_reused_references':list(REFS),'seeds':list(SEEDS),'epochs':list(EPOCHS),'sessions':list(SESSIONS),'artifact_evidence':evidence,'per_cell_operational_costs':{k:v['operational_cost'] for k,v in evidence.items() if 'operational_cost'in v},'absolute_r2':absolute,'contrasts':contrasts,'primary_oriented_contrasts':primary,'component_sufficiency':suff,'conditional_row_shuffle_trigger_arms':[x for x in suff if suff[x]['sufficient']],'analysis_notes':{'reference_policy':'qualified SHA-locked T4/TS4 only','row_shuffle':'not launched; trigger is sufficient PH4/AC4/MB4/B4'}}
 a.out.parent.mkdir(parents=True,exist_ok=True);fd=os.open(a.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'w') as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
if __name__=='__main__':main()
