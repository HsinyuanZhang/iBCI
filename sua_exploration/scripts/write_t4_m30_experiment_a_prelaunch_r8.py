#!/usr/bin/env python3
"""Write-once CPU-only r8 recovery receipt; no launch logic."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SUA=ROOT/'sua_exploration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 base=json.loads((SUA/'results/t4_m30_experiment_a_descriptor_prelaunch_v3_r5_20260802/receipt.json').read_text())
 files=['aggregate_t4_m30_experiment_a_r8.py','eval_t4_m30_experiment_a.py','eval_epoch_window_generic_dandi688.py','select_gradient_free_protocol_dandi688.py','run_t4_m30_experiment_a_r8_one_cell.sh','schedule_t4_m30_experiment_a_r8_2gpu.sh','verify_t4_m30_experiment_a_r8_authorization.py','t4_m30_experiment_a_r4_authorization.py','t4_m30_experiment_a_r5_authorization.py','t4_m30_experiment_a_r8_authorization.py','write_t4_m30_experiment_a_prelaunch_r8.py','train_variant_dandi688.py','eval_adaptation_dandi688.py']
 sources={f'sua_exploration/scripts/{x}':sha(SUA/'scripts'/x) for x in files}
 for rel in ['sua_exploration/mc_maze/unit_side_features.py','sua_exploration/mc_maze/multisession_datamodule.py','streaming_calibration_exp/src/models/components/streaming_encoders.py','streaming_calibration_exp/src/models/streaming_calibration_module.py','sua_exploration/tests/test_t4_m30_experiment_a_r8_recovery.py','sua_exploration/tests/test_t4_m30_experiment_a_r5_disk_fixture.py','sua_exploration/configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem']:
  sources[rel]=sha(ROOT/rel)
 pub=SUA/'configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem';need='ff9d1b2b985c9cd8c2697cfb0e1e5353b8c19d1f3ff094d2987aa1537c79a375'
 if sha(pub)!=need:raise ValueError('public key drift')
 base.update({'schema_version':8,'status':'PASS','source_sha256':sources,'execution':{'gpu_used':False,'training_started':False,'formal_sua_opened':False},'authorization':{'gpu_launch_authorized':False,'trust_anchor':{'algorithm':'Ed25519','public_key_path':str(pub.resolve()),'public_key_sha256':sha(pub),'fixed_authorization_path':str((a.output_dir/'root_gpu_authorization.json').resolve()),'fixed_signature_path':str((a.output_dir/'root_gpu_authorization.sig').resolve())}},'matrix':{**base['matrix'],'cache_dir':'sua_exploration/cache/t4_m30_experiment_a_v8','runtime_screen':'sua_t4_m30_component_attribution_v8','managed_execution':'managed foreground exec session only; no nohup/disown/short-lived wrapper'},'recovery_incidents':{'r5':{'all_cells_exit_code':126,'gpu_used':False,'checkpoints_created':False},'r6':{'process_supervision':'short-lived nohup wrapper reaped child tree','gpu_used':False,'checkpoints_created':False,'results_created':False}},'r8_execution_contract':{'scheduler':'schedule_t4_m30_experiment_a_r8_2gpu.sh','runner_invocation':'bash "$RUNNER"','status_hashes':['result_sha256','metadata_sha256','cost_sha256'],'pre_aggregate_sweep':'all 18 status receipts completed with exit_code 0'}})
 a.output_dir.mkdir(parents=True);(a.output_dir/'receipt.json').write_text(json.dumps(base,indent=2,sort_keys=True)+'\n');(a.output_dir/'RECEIPT.md').write_text('# Experiment A r8 recovery prelaunch\n\nStatus: **PASS (CPU-only prelaunch)**. GPU authorization remains false. Launch only from a managed foreground exec session.\n')
if __name__=='__main__':main()
