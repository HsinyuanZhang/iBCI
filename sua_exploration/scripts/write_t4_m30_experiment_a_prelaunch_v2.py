#!/usr/bin/env python3
"""Write-once v2 NO-GO/PASS seal for Experiment-A launch infrastructure; never launches."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parents[2]; S=R/'sua_exploration'; M=S/'configs/subc_co_27_6_strict_train_val_manifest.json'
def h(p):
 d=hashlib.sha256();d.update(p.read_bytes());return d.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 m=json.loads(M.read_text());sys.path.insert(0,str(S));from mc_maze.unit_side_features import compute_unit_side_features_uncached
 rows={}
 for n in m['session_splits']['train']:
  f=S/'data/dandi_000688/sub-C'/f'{n}_behavior+ecephys.nwb';_,meta=compute_unit_side_features_uncached(f,feature_group='ph4',pool_size=30);rows[n]=meta.exact_zero_m_unit_count
 files=['run_t4_m30_experiment_a_one_cell.sh','schedule_t4_m30_experiment_a_2gpu.sh','eval_t4_m30_experiment_a.py','aggregate_t4_m30_experiment_a.py','write_t4_m30_experiment_a_prelaunch_v2.py']
 sources={f'sua_exploration/scripts/{x}':h(S/'scripts'/x) for x in files};sources['sua_exploration/scripts/train_variant_dandi688.py']=h(S/'scripts/train_variant_dandi688.py');sources['sua_exploration/scripts/eval_adaptation_dandi688.py']=h(S/'scripts/eval_adaptation_dandi688.py');sources['streaming_calibration_exp/src/models/streaming_calibration_module.py']=h(R/'streaming_calibration_exp/src/models/streaming_calibration_module.py')
 blockers=['aggregate lacks required hierarchical bootstrap, exact paired Wilcoxon, paired/unpaired SE correlation, and qualified-T4 sufficiency comparison','v2 receipt has no exact static B3S cost_profile/online-cost measurement binding','post-run cost receipt is necessarily pending because no model has run']
 out={'schema_version':2,'status':'NO-GO','execution':{'gpu_used':False,'training_started':False,'formal_sua_opened':False,'source_only_ph4_counting':True},'m30_v2_receipt':{'path':'sua_exploration/results/t4_m30_experiment_a_cpu_preflight_v2_20260802/receipt.json','sha256':h(S/'results/t4_m30_experiment_a_cpu_preflight_v2_20260802/receipt.json')},'source_sha256':sources,'ph4_exact_zero_m_rows':{'by_source_session':rows,'total':sum(rows.values())},'sealed_matrix':{'arms':['Z4','PH4','AC4','MB4','B4','LS4'],'seeds':[42,43,44],'cells':18,'support_pool':30,'epochs':12,'residual_mode':'none','authorization_required':True},'authorization':{'gpu_launch_authorized':False},'blockers':blockers}
 a.output_dir.mkdir(parents=True);(a.output_dir/'receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');(a.output_dir/'RECEIPT.md').write_text('# Experiment A descriptor prelaunch v2\n\nStatus: **NO-GO**. See receipt.json blockers.\n')
if __name__=='__main__':main()
