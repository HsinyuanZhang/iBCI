#!/usr/bin/env python3
"""Formal, CPU-only M1 trained RIFT runtime benchmark; rejects incomplete runs."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,platform,resource
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]; V1=ROOT.parent/'btransform_unified_v1'; sys.path[:0]=[str(ROOT/'src'),str(V1/'src'),str(V1/'scripts'),str(ROOT.parent)]
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v2 import RiftDecoder,RiftStreamDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
import m1_train

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def cpu_model():
 for line in Path('/proc/cpuinfo').read_text().splitlines():
  if line.startswith('model name'): return line.split(':',1)[1].strip()
 return platform.processor()
def rss():
 cur=hwm=0
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'): cur=int(line.split()[1])*1024
  if line.startswith('VmHWM:'): hwm=int(line.split()[1])*1024
 return cur,hwm
def summary(v):
 a=np.array(v)*1e3; return {'n':len(v),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'mean_ms':float(a.mean())}
def load(run):
 meta=json.loads((run/'run_meta.json').read_text()); receipt=json.loads((run/'score_receipt.json').read_text())
 if meta.get('schema')!='m1_rift_train_v2' or meta.get('status')!='FORMAL' or receipt.get('status')!='COMPLETED' or receipt.get('cell')!=meta.get('cell'): raise RuntimeError('requires completed formal M1 projadd run and matching score receipt')
 epoch=int(receipt['selection']['epoch']); ck=run/f'epoch_{epoch:03d}.pt'; state=torch.load(ck,map_location='cpu',weights_only=False)
 if state.get('smoke') or state.get('cell')!=meta['cell'] or sha(ck)!=receipt['checkpoint_sha256_by_epoch'][str(epoch)]: raise RuntimeError('selected checkpoint receipt mismatch')
 model=RiftDecoder('m1',context_bins=100,bias_mode='recency',seed=42,proj_dim=16).eval(); model.load_state_dict(state['raw_state_dict']); ema=DecoderEMA(model,decay=.9995); ema.load_state_dict(state['ema']); ema.apply_to(model); return model,meta,receipt,ck,epoch
def parity(model,banks,x):
 ids=[f'i{i}' for i in range(len(banks))]; canonical=ids[:]; pos={k:i for i,k in enumerate(canonical)}; bank_by_id={i:b for i,b in zip(ids,banks)}; ref=RiftStreamDecoder(model); cached=CpuRiftRuntime(model,banks,ids,temporal_backend='cached'); mx=0.
 with torch.inference_mode():
  for t,row0 in enumerate(x):
   order=ids[::-1] if t==11 else ids
   row=np.stack([row0[pos[i]] for i in order]); inp=torch.from_numpy(row.copy())
   want=ref.stream_step(inp,[bank_by_id[i] for i in order],order); got=cached.advance(inp,order if t==11 else None); mx=max(mx,float((want-got).abs().max())); ids=order
  cached.reset_rows([ids[-1]]); ref.reset(ids[-1]); row=np.stack([x[-1,pos[i]] for i in ids]); inp=torch.from_numpy(row.copy()); want=ref.stream_step(inp,[bank_by_id[i] for i in ids],ids); got=cached.advance(inp); mx=max(mx,float((want-got).abs().max())); torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-5)
 if mx > 1e-5: raise RuntimeError(f'cached/reference parity max_abs {mx} exceeds 1e-5')
 return {'passed':True,'checked_advances':len(x)+1,'max_abs_error':mx,'threshold':1e-5,'stream_reorder_and_async_reset':True}
def fullwindow_oracle(model,banks,x):
 # Independent full W100 forward checks on a single canonical stream.
 rt=CpuRiftRuntime(model,[banks[0]],['i0'],temporal_backend='cached'); checks=[]
 with torch.inference_mode():
  for t,row in enumerate(x[:,0]):
   got=rt.advance(torch.from_numpy(row[None].copy()))
   if t in (99,200,400):
    want=model(torch.from_numpy(x[t-99:t+1,0][None].copy()),banks[0],input_valid_mask=torch.ones(1,100,dtype=torch.bool)); d=float((got-want).abs().max()); torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-5);checks.append({'end_bin':t,'max_abs_error':d})
 return {'passed':True,'checks':checks}
@torch.inference_mode()
def measure(model,banks,x,warm=300,steps=100):
 out=[]; rounds=[]
 for _ in range(3):
  rt=CpuRiftRuntime(model,banks,[f'i{i}' for i in range(len(banks))],temporal_backend='cached'); t=time.perf_counter(); rt.advance(torch.from_numpy(x[0])); startup=time.perf_counter()-t
  for v in x[1:warm]:rt.advance(torch.from_numpy(v))
  z=[]
  for v in x[warm:warm+steps]: t=time.perf_counter();rt.advance(torch.from_numpy(v));z.append(time.perf_counter()-t)
  out+=z;rounds.append({'startup_first_predict_s':startup,'steady':summary(z)})
 return {'per_round':rounds,'decode_call':summary(out),'current_rss_bytes':rss()[0],'max_rss_bytes':max(rss()[1],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),'logical_kv_bytes':None,'logical_kv_note':'not separately introspected; RSS already includes runtime KV and is not summed with it'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);m,meta,score,ck,e=load(a.run_dir.resolve()); ds,samp=m1_train.legacy.build_fullsession_face(); banks,_=m1_train.legacy.build_fullsession_banks(ds); cases={}
 for B in (1,8):
  streams=[]
  for j in range(B):
   s=m1_train.SOURCE_SESSIONS[j % len(m1_train.SOURCE_SESSIONS)]; offset=(j//len(m1_train.SOURCE_SESSIONS))*401; streams.append((s,offset,np.asarray(ds.neural_data[s][99+offset:99+offset+401],np.float32)))
  sessions=[z[0] for z in streams]; raw=np.stack([z[2] for z in streams],axis=1); assert raw.shape==(401,B,64),raw.shape
  cases[f'B{B}']={'raw_flow':{'sessions':sessions,'offsets':[z[1] for z in streams],'shape':list(raw.shape),'bins':401,'true_zero_bin_count':int(np.sum(np.all(raw==0,axis=-1))),'source':'actual FalconDataset neural_data after explicit 99-bin query-padding removal','raw_sha256':ah(raw),'bank_e0_sha256':[str(banks[s].calibration_meta['array_sha256']) for s in sessions],'bank_carrier_sha256':[str(banks[s].calibration_meta['carrier_sha256']) for s in sessions]},'parity':parity(m,[banks[s] for s in sessions],raw),'fullwindow_oracle':fullwindow_oracle(m,[banks[s] for s in sessions],raw),'cached':measure(m,[banks[s] for s in sessions],raw)}
 rep={'schema':'m1_r100_trained_rift_cpu_v1','device':'cpu','threads':{'intra':2,'interop':1},'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'run_meta_sha256':sha(a.run_dir/'run_meta.json'),'score_receipt_sha256':sha(a.run_dir/'score_receipt.json'),'selected_ema_epoch':e,'weights':'EMA from selected formal checkpoint','protocol':{'warmup':300,'rounds':3,'measured_calls_per_round':100,'advances':401,'no_onnx':True},'parameters':sum(x.numel() for x in m.parameters()),'environment':{'cpu':cpu_model(),'torch':torch.__version__,'affinity':sorted(os.sched_getaffinity(0)),'inference_mode':True},'script_sha256':sha(Path(__file__)),'cases':cases};a.dest.mkdir(parents=True,exist_ok=False);(a.dest/'benchmark.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
