#!/usr/bin/env python3
"""Write fail-closed provenance for one native-MUA held-out test-only cell."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
from src.data.falcon_datamodule import FalconDataModule
from src.data.falcon_t4_features import calibration_target_angles, _design, validate_trial_label_alignment
from falcon_challenge.config import FalconConfig, FalconTask

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--artifact',type=Path,required=True);p.add_argument('--source-artifact',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--task',choices=('m1','m2'),required=True);p.add_argument('--group',choices=('f0','t4','ts4'),required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--support',type=int,required=True);p.add_argument('--frozen-aggregate',type=Path,required=True);a=p.parse_args()
 r=a.artifact/'resolved_config.yaml'; s=a.artifact/'split_manifest.json'; m=a.artifact/'metrics_per_session.csv'
 if not all(x.is_file() for x in (r,s,m)): raise ValueError(f'incomplete test artifact {a.artifact}')
 split=json.loads(s.read_text()); resolved=OmegaConf.to_container(OmegaConf.load(r),resolve=False)
 if split.get('heldout_evaluated_in_fit') or not split.get('heldout_evaluated_in_test'): raise ValueError('heldout gate invalid')
 data=resolved.get('data',{}); model=resolved.get('model',{})
 expected_group={'f0':'none','t4':'t4','ts4':'ts4'}[a.group]
 if data.get('task')!=a.task or model.get('variant') not in ({'f0':'B3','t4':'B3S','ts4':'B3S'}[a.group],) or data.get('side_feature_group')!=expected_group or data.get('random_calibration') is not False or resolved.get('train') is not False or resolved.get('test') is not True: raise ValueError('resolved test-only contract mismatch')
 frozen=json.loads(a.frozen_aggregate.read_text()); key=f'fold{a.fold}_seed{a.seed}'
 if Path(frozen['artifacts'][a.task][a.group].get(key,'')).resolve()!=a.source_artifact.resolve(): raise ValueError('source artifact is not frozen aggregate mapping')
 source_meta=json.loads((a.source_artifact/'run_metadata.json').read_text()); source_split=json.loads((a.source_artifact/'split_manifest.json').read_text()); source_ckpt=json.loads((a.source_artifact/'checkpoint_manifest.json').read_text())
 if Path(source_ckpt['artifact_checkpoint_path']).resolve()!=a.checkpoint.resolve() or source_ckpt['artifact_checkpoint_sha256']!=sha(a.checkpoint): raise ValueError('source checkpoint manifest mismatch')
 if source_meta.get('seed')!=a.seed or source_meta.get('fold_id')!=a.fold or source_split.get('train_sessions')!=split.get('train_sessions'): raise ValueError('source fold/seed/train split mismatch')
 if expected_group!='none' and split.get('native_t4_normalization',{}).get('train_sessions')!=source_split.get('train_sessions'): raise ValueError('normalization not fit on source fold train sessions')
 data_dir=Path('/home/xinyuan/Work_host/SPINT/SPINT-main/data')/('000941' if a.task=='m1' else '000953')
 audits=[]
 for f in sorted(data_dir.rglob('*held-out-calib*.nwb')):
  task_enum=FalconConfig(task=FalconTask.__dict__[a.task]).task; angles=calibration_target_angles(f,a.task); raw=FalconDataModule(task=a.task,data_dir=str(data_dir)).load_data(f,task_enum,True); filtered=FalconDataModule(task=a.task,data_dir=str(data_dir)).load_data(f,task_enum,bool(data.get('use_intertrials',True))); validate_trial_label_alignment(raw[2],angles,source=str(f)); kept=angles if data.get('use_intertrials',True) else angles[np.asarray(raw[3],bool)[np.flatnonzero(raw[2])]]; validate_trial_label_alignment(filtered[2],kept,source=str(f)); angles=kept[:a.support]; d,valid=_design(angles); rank=int(np.linalg.matrix_rank(d)); cond=float(np.linalg.cond(d)) if rank==3 else float('inf')
  if rank!=3: raise ValueError(f'rank failure {f}')
  audits.append({'session':f.name.split('_')[1].split('.')[0],'calibration_nwb':str(f),'prefix_trials':a.support,'directional_trials':int(valid.sum()),'rank':rank,'condition_number':cond})
 out={'schema_version':1,'task':a.task,'group':a.group,'fold':a.fold,'seed':a.seed,'train':False,'test':True,'uses_backward_gradients':False,'source_artifact':str(a.source_artifact.resolve()),'source_checkpoint':str(a.checkpoint.resolve()),'source_checkpoint_sha256':sha(a.checkpoint),'heldout_calibration_audit':audits,'support_trials':a.support,'train_only_normalization':split.get('native_t4_normalization'),'resolved_config':str(r.resolve()),'split_manifest':str(s.resolve())}
 (a.artifact/'heldout_t4_provenance.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
