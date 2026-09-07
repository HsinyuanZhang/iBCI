"""Frozen e8/SPINT replay on the four existing dual-track ext4 compact banks.

Development comparison only: original SPINT training monitored heldout R2.
"""
from __future__ import annotations
import importlib.util,json,os,sys,hashlib
from pathlib import Path
import numpy as np, torch
from tfpd_exploration.src.m2_dual_track_v1 import data,plan
from tfpd_exploration.src.m2_same_query_comparator_v1 import core,physical
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'tfpd_exploration/results/m2/family_v1/ext4_e8_spint_dev_replay_v1.json'; ENV='M2_FAMILY_V1_EXT4_DEV_REPLAY'
PAY=ROOT/'tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl'
def mod():
 p=ROOT/'tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py'; s=importlib.util.spec_from_file_location('ext4_e8_payload',p); m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def run():
 if os.environ.get(ENV)!='1':raise RuntimeError('refused')
 if OUT.exists():raise RuntimeError('overwrite')
 from sua_exploration.evalai_t4_m2.export_t4_payload import calibration_file_map,load_frozen_model_and_data
 e=mod();pay=e.load_payload(PAY);model,dm,task,meta=load_frozen_model_and_data(); tags=calibration_file_map(ROOT/'SPINT-main/data/000953',task); dec=e.build_decoder('small');dec.load_state_dict({k:torch.as_tensor(v,dtype=torch.float32) for k,v in pay['state_dict'].items()});dec.cuda().eval();teacher=model.teacher.cuda().eval();rows=[]; eall=[];sall=[];yall=[]
 for ses in plan.EXT4_SESSIONS:
  bank=data.load_session_bank('ext4',ses,device='cuda'); X=np.asarray(bank.X_store,dtype=np.float32);y=np.asarray(bank.target_store,dtype=np.float32);starts=np.asarray(bank.eligible_starts,dtype=np.int64); tag=tags[ses];pb=pay['bank_by_dataset_tag'][tag]; eb=e.SessionBank(torch.as_tensor(pb['E0'],device='cuda'),torch.as_tensor(pb['T'],device='cuda'),torch.as_tensor(pb['unit_mask'],device='cuda'))
  act=torch.from_numpy(np.asarray(data.read_memmap(data._session_dir('ext4',ses)/'calib_activity.npy'),dtype=np.float32)[:30]).unsqueeze(0).cuda()
  with torch.inference_mode(): sid=physical._teacher_identity(torch,teacher,act)
  ep=[];sp=[]
  for i in range(0,len(starts),128):
   ss=starts[i:i+128]; windows=X[i:i+len(ss)] if X.ndim==3 else np.stack([X[int(v):int(v)+50] for v in ss]); z=torch.from_numpy(np.ascontiguousarray(windows)).cuda()
   with torch.inference_mode(): ep.append((dec.forward_last(z,eb)/5).cpu().numpy());sp.append((physical._manual_teacher_decode(teacher,z,sid)[:,-1,:]/5).cpu().numpy())
  ep=np.concatenate(ep);sp=np.concatenate(sp); eall.append(ep);sall.append(sp);yall.append(y); h=core.array_sha256(starts);t=core.array_sha256(y);rows.append({'session':ses,'window_count':len(starts),'ordered_window_starts_sha256':h,'target_sha256':t,'e8_prediction_sha256':core.array_sha256(ep),'spint_prediction_sha256':core.array_sha256(sp),'e8_r2':core.variance_weighted_r2(y,ep),'spint_r2':core.variance_weighted_r2(y,sp),'native_units':True,'same_starts_targets':True})
 out={'schema':'m2_family_v1_ext4_e8_spint_dev_replay_v1','status':'DEVELOPMENT_COMPARISON_NOT_UNBIASED_HELDOUT','warning':'BOTH historical systems have development/selection exposure: original SPINT monitored val_heldout/r2_mean and e8 ext6 package records external epoch-pick score; never select new points here','sessions':list(plan.EXT4_SESSIONS),'rows':rows,'e8_equal_session_r2':float(np.mean([r['e8_r2'] for r in rows])),'spint_equal_session_r2':float(np.mean([r['spint_r2'] for r in rows])),'e8_pooled_r2':core.variance_weighted_r2(np.concatenate(yall),np.concatenate(eall)),'spint_pooled_r2':core.variance_weighted_r2(np.concatenate(yall),np.concatenate(sall)),'e8_payload_sha256':hashlib.sha256(PAY.read_bytes()).hexdigest(),'spint_teacher_sha256':meta['teacher_checkpoint_sha256'],'ext4_cache_root':str(data.cache_root()/'ext4')}
 body=(json.dumps(out,indent=2,sort_keys=True)+'\n').encode();OUT.write_bytes(body);OUT.with_suffix('.json.sha256').write_text(hashlib.sha256(body).hexdigest()+'  '+OUT.name+'\n');OUT.chmod(0o444);return out
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
