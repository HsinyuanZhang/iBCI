#!/usr/bin/env python3
"""Frozen e3 M1 D-JOINT carrier-column input probe on public HO3 M10."""
from __future__ import annotations
import argparse,dataclasses,hashlib,json,os,sys
from pathlib import Path
from typing import Any
import numpy as np,torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;V2=HERE.parents[1];WS=V2.parent
for p in (V2/'src',V2/'scripts',WS,WS/'btransform_unified_v1'/'src',WS/'btransform_unified_v1'/'scripts'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.r2 import variance_weighted_r2
import rift_v1.m1_joint_train as base
RUN=V2/'results/rift_v1/m1_r100_joint_d_s42_formal_v1'; REPLAY=V2/'results/m1_muscle_r100_v1/baseline_replay/epoch_003'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def arm_carrier(x,name):
 y=np.ascontiguousarray(x,np.float32).copy()
 if name=='zero_intercept_column4':y[:,3]=0
 elif name=='zero_slopes_first3':y[:,:3]=0
 elif name=='all_zero':y[:]=0
 return y
def score(model,material,device,out):
 rows={}
 for s in base.HO:
  item=material[s];ps=[];ys=[]
  for bi,b in enumerate(DataLoader(item['dataset'],batch_size=base.BATCH,shuffle=False,num_workers=0)):
   x,y=b[0],b[1];off=bi*base.BATCH;valid=base.valid_mask_from_padded_starts(item['starts'][off:off+len(x)],device=device)
   with torch.inference_mode():z=model(x.float().to(device),item['bank'],input_valid_mask=valid)
   ps.append(np.ascontiguousarray(z.float().cpu().numpy(),np.float32));ys.append(np.ascontiguousarray(y[:,-1,:].numpy(),np.float32))
  pred=np.concatenate(ps);target=np.concatenate(ys);starts=np.asarray(item['starts'],np.int64);np.savez_compressed(out/f'{s}_predictions.npz',pred=pred,y=target,starts=starts)
  rows[s]={'legacy_r2':float(variance_weighted_r2(target,pred)),'sklearn_channel_variance_weighted_r2':float(r2_score(target,pred,multioutput='variance_weighted')),'prediction_sha256':ash(pred),'target_sha256':ash(target),'starts_sha256':ash(starts),'artifact_sha256':sha(out/f'{s}_predictions.npz'),'windows':int(len(pred))}
 return {'per_session':rows,'equal_session_mean_legacy':float(np.mean([rows[s]['legacy_r2'] for s in base.HO])),'equal_session_mean_sklearn_channel_variance_weighted':float(np.mean([rows[s]['sklearn_channel_variance_weighted_r2'] for s in base.HO]))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=4);a=p.parse_args();out=a.output_dir.resolve()
 if out.exists():raise FileExistsError(out)
 meta=json.loads((RUN/'run_meta.json').read_text());rec=json.loads((RUN/'score_receipt.json').read_text());ck=RUN/'epoch_003.pt'
 if sha(ck)!=rec['checkpoint_sha256_by_epoch']['3']:raise RuntimeError('e3 checkpoint SHA drift')
 out.mkdir(parents=True);torch.set_num_threads(a.cpu_threads);device=torch.device(a.device);material=base._ho_material();base._ho_contract(material);original_banks={s:item['bank'] for s,item in material.items()}
 result={'schema':'carrier_v4_m1_frozen_e3_input_probe_v1','status':'COMPLETED','scope':'frozen-input perturbation; public HO calibration M10/query only; not retraining or component causal attribution; not hidden evaluation','epoch':3,'baseline_run':str(RUN),'checkpoint_sha256':sha(ck),'baseline_receipt_sha256':sha(RUN/'score_receipt.json'),'code_sha256':sha(Path(__file__).resolve()),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':str(device),'arms':{}}
 state=base._validate_checkpoint(ck,meta,expected_epoch=3)
 for name in ('original_rsyn','zero_intercept_column4','zero_slopes_first3','all_zero'):
  banks={};calib={}
  for s,item in material.items():
   original=original_banks[s];t=arm_carrier(original.carrier,name);banks[s]=dataclasses.replace(original,carrier=t);calib[s]=item['calib10'];item['bank']=banks[s]
  model=base._decoder(device,'D_JOINT',base.SEED);model.install_session_memory(banks,calib);model.to(device);model.load_state_dict(state['raw_state_dict'],strict=True);ema=DecoderEMA(model,decay=base.plan.EMA_DECAY);ema.load_state_dict(state['ema']);d=out/name;d.mkdir();report=base._with_ema_eval(model,ema,lambda:score(model,material,device,d))
  if name=='original_rsyn':
   for s in base.HO:
    with np.load(REPLAY/f'{s}_predictions.npz') as z: expected=ash(z['pred'])
    if report['per_session'][s]['prediction_sha256']!=expected:raise RuntimeError(f'{s}: original prediction SHA differs from baseline replay')
  result['arms'][name]=report
 atomic(out/'receipt.json',result)
if __name__=='__main__':main()
