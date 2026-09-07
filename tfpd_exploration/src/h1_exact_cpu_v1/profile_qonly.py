"""Bounded profile for static-carrier plus final-Q-only exact E."""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from tfpd_exploration.src.h1_optimized_v2.cache import build_or_load
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.benchmark_stream import FullWindowAPI
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .profile_static import CALLS, OUT, STATE, WARMUP, model, q, r2, run, sha
from .static_frontend import StaticCarrierExactFullWindowStream
from .static_frontend_qonly import StaticCarrierQOnlyExactFullWindowStream

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    cache=build_or_load();session=next(iter(cache['minival']));row=cache['minival'][session]
    x=np.asarray(row['neural'],dtype=np.float32);target=np.asarray(row['velocity'][WARMUP:WARMUP+CALLS],dtype=np.float32)
    bank=H1Bank(*[row['bank'][k] for k in ('E0','T','unit_mask')])
    full,pfull=run(FullWindowAPI,model(),bank,session,x)
    static,pstatic=run(StaticCarrierExactFullWindowStream,model(),bank,session,x)
    qonly,pqonly=run(StaticCarrierQOnlyExactFullWindowStream,model(),bank,session,x)
    d={"schema":"h1_exact_cpu_v1_static_carrier_qonly_profile_v1","scope":"128 real-source calls only; no formal 2048 timing","affinity":sorted(os.sched_getaffinity(0)),"torch_threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),"calls":CALLS,"startup_warmup_calls":WARMUP,"session":session,"state_sha256":sha(STATE),"code_sha256":{"static_frontend_qonly.py":sha(__import__('pathlib').Path(__file__).with_name('static_frontend_qonly.py')),"profile_qonly.py":sha(__import__('pathlib').Path(__file__))},"original_full_recompute":full,"static_carrier_exact_e":static,"static_carrier_qonly_exact_e":qonly,"native_parity":{"full_vs_qonly_max_abs":float(np.max(abs(pfull-pqonly))),"static_vs_qonly_max_abs":float(np.max(abs(pstatic-pqonly))),"gate":"<=1e-5"},"r2":{"full":r2(pfull,target),"static":r2(pstatic,target),"qonly":r2(pqonly,target),"static_qonly_abs_delta":abs(r2(pstatic,target)-r2(pqonly,target)),"gate":"<=1e-5"}}
    d['native_parity']['passes']=max(d['native_parity']['full_vs_qonly_max_abs'],d['native_parity']['static_vs_qonly_max_abs'])<=1e-5;d['r2']['passes']=d['r2']['static_qonly_abs_delta']<=1e-5
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/f'profile128_threads{args.threads}_static_carrier_qonly_v1.json'
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');print(json.dumps({'path':str(p),'parity':d['native_parity'],'r2':d['r2'],'p95':{k:v['timed']['p95_ms'] for k,v in [('full',full),('static',static),('qonly',qonly)]}}))
if __name__=='__main__':main()
