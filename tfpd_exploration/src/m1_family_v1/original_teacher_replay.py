"""Frozen Original-SPINT teacher overlap diagnostic on chron80 source minival."""
from __future__ import annotations
import json,sys,hashlib
import torch
from .family_train_v2 import _source_sets,_batches,_score
from tfpd_exploration.src.m1_optimized_v2 import plan

CKPT=plan.REPO_ROOT/"streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/checkpoints/best_ckpt/epoch_018.ckpt"
def run(device="cuda:0"):
 exp=str(plan.REPO_ROOT/"streaming_calibration_exp"); sys.path.insert(0,exp) if exp not in sys.path else None
 from src.models.components.spint import SpintModel
 d=torch.load(CKPT,map_location="cpu",weights_only=False); m=SpintModel(1024,16,100,num_heads=8,num_layers=1,num_id_layers=3); m.load_state_dict({k[4:]:v for k,v in d["state_dict"].items() if k.startswith("net.")},strict=True); m.to(device)
 loaded,_tr,dev,split=_source_sets(); score=_score(m,dev,_batches(dev),torch.device(device),original_spint=True)
 out={"schema":"m1_family_v1_frozen_original_teacher_chron80_overlap_v1","status":"TRAINING_OVERLAP_DIAGNOSTIC_ONLY","checkpoint":str(CKPT),"checkpoint_sha256":hashlib.sha256(CKPT.read_bytes()).hexdigest(),"split":split,"bank_sha256":loaded["receipt"]["digests"]["npz"],"metrics":score,"outer_query_opened":False,"weights_modified":False}
 p=plan.RESULT_ROOT/"family_v1"/"original_teacher_chron80_overlap.json";p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");return out
if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
