#!/usr/bin/env python3
"""Fail closed source receipt for the seed42 M2 joint upper-bound matrix."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'streaming_calibration_exp/outputs/streaming_calibration'
ARMS={'zero4':'zero4','t4':'t4','ts4':'ts4'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def one(pat):
 xs=sorted(x for x in OUT.glob(pat) if x.is_dir())
 if len(xs)!=1: raise ValueError(f'expected one {pat}, got {xs}')
 return xs[0]
def arm(name):
 p=one(f'm2_joint_t4_upperbound_v2_{name}_f1_s42_*'); c=yaml.safe_load((p/'resolved_config.yaml').read_text()); d,m=c['data'],c['model']; tm=json.loads((p/'teacher_metadata.json').read_text())
 if (c.get('train'),c.get('test'),c.get('no_early_stopping'),c.get('require_baseline_validation'))!=(True,False,True,False): raise ValueError(f'{name}: train/selection contract')
 if (d.get('task'),d.get('loso_fold'),d.get('calibration_n_trials'),d.get('random_calibration'),d.get('include_heldout_in_fit'),d.get('include_heldout_in_test'),d.get('query_start_trial'),d.get('side_feature_group'))!=('m2',1,24,False,False,False,0,ARMS[name]): raise ValueError(f'{name}: source data contract')
 if (m.get('variant'),m.get('side_dim'),m.get('freeze_decoder'),m.get('require_clean_teacher_receipt'))!=('B3S',4,False,True): raise ValueError(f'{name}: joint B3S/decoder contract')
 logs=sorted((ROOT/'streaming_calibration_exp/logs/train/runs').glob(f'*rid-m2_joint_t4_upperbound_v2_{name}_f1_s42'))
 if len(logs)!=1: raise ValueError(f'{name}: expected one fixed-epoch log run, got {logs}')
 cp=logs[0]/'checkpoints/best_ckpt/last.ckpt'
 if c.get('trainer',{}).get('max_epochs')!=12 or not cp.is_file(): raise ValueError(f'{name}: fixed final epoch checkpoint missing')
 receipt=Path(str(m.get('teacher_receipt_path',''))).resolve()
 if not receipt.is_file(): raise ValueError(f'{name}: missing teacher receipt')
 if tm.get('teacher_checkpoint_sha256') != sha(Path(str(m['teacher_ckpt_path']))): raise ValueError(f'{name}: teacher checkpoint receipt drift')
 # The checkpoint hash is recorded in the receipt before this trusted-local
 # read; Lightning serializes OmegaConf metadata, so weights_only cannot
 # enumerate its state_dict on this environment.
 state=torch.load(cp,map_location='cpu',weights_only=False).get('state_dict',{})
 dec=[k for k in state if str(k).startswith('student.decoder.')]
 enc=[k for k in state if str(k).startswith('student.id_encoder.')]
 if not dec or not enc: raise ValueError(f'{name}: final checkpoint lacks decoder/encoder state')
 return {'artifact':str(p.resolve()),'checkpoint':{'path':str(cp.resolve()),'sha256':sha(cp),'selection':'fixed_final_epoch_11_of_12'},'teacher':{**tm,'receipt_path':str(receipt),'receipt_sha256':sha(receipt)},'optimizer_trainable_provenance':{'freeze_decoder':False,'runtime_assertion_source':str((ROOT/'streaming_calibration_exp/src/models/streaming_calibration_module.py').resolve()),'runtime_assertion_source_sha256':sha(ROOT/'streaming_calibration_exp/src/models/streaming_calibration_module.py'),'final_state_tensor_keys':{'decoder':len(dec),'encoder':len(enc)},'contract':'runtime requires all decoder+encoder parameters trainable and optimizer set exactly equals their union'},'source_manifest_sha256':sha(p/'source_manifest.json')}
def main():
 q=argparse.ArgumentParser(); q.add_argument('--out',type=Path,required=True); a=q.parse_args()
 if a.out.exists(): raise FileExistsError(a.out)
 z={x:arm(x) for x in ARMS}; teachers={json.dumps(v['teacher'],sort_keys=True) for v in z.values()}
 if len(teachers)!=1: raise ValueError('teacher init differs across zero4/T4/TS4')
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'schema_version':1,'screen':'m2_joint_t4_upperbound_v2','seed':42,'teacher_init_identical':True,'source_heldout_opened':False,'arms':z,'status':'source_only_complete__heldout_replay_requires_separate_fail_closed_runner'},indent=2,sort_keys=True)+'\n'); print(a.out)
if __name__=='__main__': main()
