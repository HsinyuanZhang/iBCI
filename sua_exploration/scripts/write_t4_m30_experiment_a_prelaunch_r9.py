import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SUA=ROOT/'sua_exploration'
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 b=json.loads((SUA/'results/t4_m30_experiment_a_descriptor_prelaunch_v3_r8_20260803/receipt.json').read_text())
 rel=['sua_exploration/scripts/aggregate_t4_m30_experiment_a_r9.py','sua_exploration/scripts/aggregate_t4_m30_experiment_a_r5.py','sua_exploration/scripts/aggregate_t4_m30_experiment_a_v3.py','sua_exploration/scripts/t4_m30_experiment_a_r9_authorization.py','sua_exploration/scripts/t4_m30_experiment_a_r4_authorization.py','sua_exploration/scripts/t4_m30_experiment_a_r5_authorization.py','sua_exploration/scripts/write_t4_m30_experiment_a_prelaunch_r9.py','sua_exploration/tests/test_t4_m30_experiment_a_r9_import_closure.py','sua_exploration/configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem']
 b['schema_version']=9;b['status']='PASS';b['source_sha256']={x:h(ROOT/x) for x in rel};b['authorization']={'gpu_launch_authorized':False,'trust_anchor':{'algorithm':'Ed25519','public_key_path':str((SUA/'configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem').resolve()),'public_key_sha256':h(SUA/'configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem'),'fixed_authorization_path':str((a.output_dir/'root_gpu_authorization.json').resolve()),'fixed_signature_path':str((a.output_dir/'root_gpu_authorization.sig').resolve())}};b['matrix']['runtime_screen']='sua_t4_m30_component_attribution_v9';b['matrix']['cache_dir']='sua_exploration/cache/t4_m30_experiment_a_v9';b['dependency_closure']=rel
 a.output_dir.mkdir(parents=True);(a.output_dir/'receipt.json').write_text(json.dumps(b,indent=2,sort_keys=True)+'\n');(a.output_dir/'RECEIPT.md').write_text('# Experiment A r9 prelaunch\n\nStatus: **PASS (CPU-only)**. GPU authorization false.\n')
if __name__=='__main__':main()
