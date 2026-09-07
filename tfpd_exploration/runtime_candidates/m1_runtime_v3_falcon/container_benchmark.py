"""Local-only container public API benchmark using frozen source fixture."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/')
import torch
from falcon_challenge.config import FalconConfig,FalconTask
from m1_trf_falcon_decoder import M1TemporalFalconDecoder
def read(path):
 try:return int(Path(path).read_text().strip())
 except OSError:return None
def main(a):
 z=np.load('/work/fixtures/source_b4_public_parity.npz',allow_pickle=False);x=z['inputs'];stems=[str(q) for q in z['stems']]
 if hashlib.sha256(Path('/data/decoder.pkl').read_bytes()).hexdigest()!=str(z['payload_sha256'][0]):raise RuntimeError('payload')
 torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 t=time.perf_counter_ns();d=M1TemporalFalconDecoder(FalconConfig(task=FalconTask.m1),'/data/decoder.pkl',batch_size=4);d.reset(stems);construct=(time.perf_counter_ns()-t)/1e6
 t=time.perf_counter_ns();y=d.predict(np.ascontiguousarray(x[0]));first=(time.perf_counter_ns()-t)/1e6
 if y.shape!=(4,16) or y.dtype!=np.float32 or (not a.allow_view and not y.flags.owndata):raise RuntimeError('public')
 for i in range(a.warmup): d.predict(np.ascontiguousarray(x[(i+1)%len(x)]))
 values=[];rss0=read('/sys/fs/cgroup/memory.current');peak=rss0
 for i in range(a.calls):
  t=time.perf_counter_ns();y=d.predict(np.ascontiguousarray(x[(i+a.warmup+1)%len(x)]));values.append((time.perf_counter_ns()-t)/1e6);peak=max(peak or 0,read('/sys/fs/cgroup/memory.current') or 0)
  if y.shape!=(4,16) or y.dtype!=np.float32 or (not a.allow_view and not y.flags.owndata) or not np.isfinite(y).all():raise RuntimeError('public drift')
 runtime_path=Path('/v3_runtime.py')
 out={'schema':'m1_v3_local_container_public_bench_v1','calls':a.calls,'warmup':a.warmup,'construct_reset_ms':construct,'first_public_return_ms':first,'public_b4_ms':{f'p{q}':float(np.percentile(values,q)) for q in (50,95,99)}|{'mean':float(np.mean(values)),'max':float(max(values))},'container_cgroup_memory_before_bytes':rss0,'container_cgroup_memory_peak_bytes':peak,'torch':torch.__version__,'torch_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0)),'payload_sha256':str(z['payload_sha256'][0]),'candidate_decoder_sha256':hashlib.sha256(Path('/m1_trf_falcon_decoder.py').read_bytes()).hexdigest(),'candidate_runtime_sha256':hashlib.sha256(runtime_path.read_bytes()).hexdigest() if runtime_path.is_file() else None,'local_only':True}
 out['output_is_owning_copy_required']=not a.allow_view
 Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--calls',type=int,required=True);p.add_argument('--warmup',type=int,default=128);p.add_argument('--output',default='/work/container_benchmark.json');p.add_argument('--allow-view',action='store_true');main(p.parse_args())
