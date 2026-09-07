"""Compare original FULL, existing E, and isolated static-carrier E at 128 calls."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load
from tfpd_exploration.src.h1_optimized_v4.model import H1SignedFull
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.benchmark_stream import FullWindowAPI
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import ExactFullWindowStream
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .static_frontend import StaticCarrierExactFullWindowStream

OUT = ROOT / "exact_cpu_v1"
STATE = ROOT / "paired_v4_signed_12ep_v1/independent_score_export/full_epoch12_plain_ema_model_state.pt"
CALLS, WARMUP = 128, 50

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def q(values):
    x=np.asarray(values,dtype=np.float64)
    return {"p50_ms":float(np.percentile(x,50)),"p95_ms":float(np.percentile(x,95)),"p99_ms":float(np.percentile(x,99)),"mean_ms":float(x.mean())}
def r2(p,y): return float(1-np.square(p-y).sum()/np.square(y-y.mean(0,keepdims=True)).sum())
def model():
    m=H1SignedFull().eval();m.load_state_dict(torch.load(STATE,map_location="cpu",weights_only=True),strict=True);return m
def run(cls, model, bank, session, x):
    t=time.perf_counter_ns();e=cls(model,bank,task="h1",session_id=session,unit_ids=range(x.shape[1]));ctor=(time.perf_counter_ns()-t)/1e6
    t=time.perf_counter_ns();e.reset();reset=(time.perf_counter_ns()-t)/1e6
    startup=[]
    for i in range(WARMUP):
        t=time.perf_counter_ns();e.predict(x[i][None]);startup.append((time.perf_counter_ns()-t)/1e6)
    timing=[];out=[]
    for i in range(CALLS):
        t=time.perf_counter_ns();out.append(e.predict(x[WARMUP+i][None])[0]);timing.append((time.perf_counter_ns()-t)/1e6)
    return {"ctor_ms":ctor,"reset_ms":reset,"startup_first_ms":startup[0],"startup_p50_ms":float(np.median(startup)),"timed":q(timing)},np.asarray(out,dtype=np.float32)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=2);args=parser.parse_args()
    if args.threads<1:raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    cache=build_or_load();session=next(iter(cache['minival']));row=cache['minival'][session];x=np.asarray(row['neural'],dtype=np.float32)
    bank=H1Bank(*[row['bank'][k] for k in ('E0','T','unit_mask')]);target=np.asarray(row['velocity'][WARMUP:WARMUP+CALLS],dtype=np.float32)
    full,pfull=run(FullWindowAPI,model(),bank,session,x)
    exact,pexact=run(ExactFullWindowStream,model(),bank,session,x)
    static,pstatic=run(StaticCarrierExactFullWindowStream,model(),bank,session,x)
    receipt={"schema":"h1_exact_cpu_v1_static_carrier_profile_v1","scope":"128 real-source calls only; no 2048 formal timing","affinity":sorted(os.sched_getaffinity(0)),"torch_threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),"platform":platform.platform(),"calls":CALLS,"startup_warmup_calls":WARMUP,"session":session,"state_sha256":sha(STATE),"code_sha256":{"static_frontend.py":sha(Path(__file__).with_name('static_frontend.py')),"profile_static.py":sha(Path(__file__))},"measurement":"whole public native API wall; B1 only","original_full_recompute":full,"existing_exact_e":exact,"static_carrier_exact_e":static,"native_parity":{"original_vs_existing_max_abs":float(np.max(abs(pfull-pexact))),"existing_vs_static_max_abs":float(np.max(abs(pexact-pstatic))),"gate":"<=1e-5"},"r2":{"original":r2(pfull,target),"existing":r2(pexact,target),"static":r2(pstatic,target),"existing_static_abs_delta":abs(r2(pexact,target)-r2(pstatic,target)),"gate":"<=1e-5"}}
    receipt['native_parity']['passes']=max(receipt['native_parity']['original_vs_existing_max_abs'],receipt['native_parity']['existing_vs_static_max_abs'])<=1e-5;receipt['r2']['passes']=receipt['r2']['existing_static_abs_delta']<=1e-5
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'profile128_threads{args.threads}_static_carrier_v1.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({"path":str(path),"parity":receipt['native_parity'],"r2":receipt['r2'],"p95":{k:v['timed']['p95_ms'] for k,v in [('full',full),('exact',exact),('static',static)]}}))
if __name__=='__main__':main()
