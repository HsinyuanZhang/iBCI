"""Read-only e8 payload replay on the exact frozen comparator query surface."""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
from tfpd_exploration.src.m2_same_query_comparator_v1 import core, physical

ROOT=Path(__file__).resolve().parents[3]
PAYLOAD=ROOT/"tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
OUT=ROOT/"tfpd_exploration/results/m2/family_v1/e8_within_post30_replay_v1.json"
ENV="M2_FAMILY_V1_E8_REPLAY"
def _sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def _module():
 p=ROOT/"tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py"; s=importlib.util.spec_from_file_location("m2_e8_frozen_payload",p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
def run():
 if os.environ.get(ENV)!="1": raise RuntimeError(f"REFUSED: {ENV}=1 required")
 if OUT.exists(): raise RuntimeError("refusing overwrite")
 import torch
 from sua_exploration.evalai_t4_m2.export_t4_payload import calibration_file_map, load_frozen_model_and_data
 m=_module(); payload=m.load_payload(PAYLOAD)
 if payload["kind"]!="small" or payload["behavior_scaling_factor"]!=5.0 or payload.get("smooth_observations") is not False: raise RuntimeError("payload runtime contract drift")
 _model,dm,task,meta=load_frozen_model_and_data(); tags=calibration_file_map(ROOT/"SPINT-main/data/000953",task)
 decoder=m.build_decoder(payload["kind"]); decoder.load_state_dict({n:torch.as_tensor(v,dtype=torch.float32) for n,v in payload["state_dict"].items()},strict=True); decoder.cuda().eval()
 rows=[]
 for session in sorted(dm.train_dataset.calib_trialized_neural_features):
  tag=tags[session]
  if tag not in payload["bank_by_dataset_tag"]: raise RuntimeError(f"e8 has no bank for {session}")
  b=payload["bank_by_dataset_tag"][tag]; bank=m.SessionBank(torch.as_tensor(b["E0"],device="cuda"),torch.as_tensor(b["T"],device="cuda"),torch.as_tensor(b["unit_mask"],device="cuda"))
  starts,target=physical._query_arrays(dm.train_dataset,session,"within_post30"); pred=[]
  for off in range(0,len(starts),256):
   ss=starts[off:off+256]; x=np.stack([dm.train_dataset.neural_data[session][z:z+50] for z in ss]).astype(np.float32,copy=False)
   with torch.inference_mode(): pred.append((decoder.forward_last(torch.from_numpy(x).cuda(),bank)/5.0).cpu().numpy().astype(np.float32))
  p=np.ascontiguousarray(np.concatenate(pred)); rows.append({"session":session,"surface":"within_post30","window_count":int(len(starts)),"ordered_window_starts_sha256":core.array_sha256(starts),"target_sha256":core.array_sha256(target),"prediction_sha256":core.array_sha256(p),"r2":core.variance_weighted_r2(target,p),"native_output_divisor":5.0,"payload_dataset_tag":tag})
 out={"schema":"m2_family_v1_e8_within_post30_replay_v1","status":"LOCAL_TRAINING_OVERLAP_DIAGNOSTIC_NOT_GENERALIZATION","payload_path":str(PAYLOAD),"payload_sha256":_sha(PAYLOAD),"payload_metadata":payload["metadata"],"spint_teacher_sha256":meta["teacher_checkpoint_sha256"],"surface":"within_post30","same_query_and_target_as_spint":True,"rows":rows,"summary":core.summarize_sessions({x["session"]:x["r2"] for x in rows})}
 body=(json.dumps(out,indent=2,sort_keys=True)+"\n").encode(); OUT.write_bytes(body); OUT.with_suffix(".json.sha256").write_text(hashlib.sha256(body).hexdigest()+"  "+OUT.name+"\n"); OUT.chmod(0o444); return out
if __name__=="__main__": print(json.dumps(run(),sort_keys=True))
