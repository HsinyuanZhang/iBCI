"""Fresh-process public-NumPy microbenchmark for the isolated M1 V3 runtime.

This deliberately benchmarks only the selected source T operator against its
matched existing cached implementation.  The submitted full-window package is
recorded elsewhere as a known separate baseline and is not conflated with this
operator-equivalence comparison.
"""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch

from .runtime import BankBatch, HeterogeneousCurrentQueryStream

ROOT=Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1")
POST=ROOT/"formal_formal12_chron80_v2_current_query_postscore.json"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"): return int(line.split()[1])*1024
    return None
def qs(xs): return {f"p{q}_ms":float(np.percentile(xs,q)) for q in (50,95,99)}|{"mean_ms":float(np.mean(xs)),"max_ms":float(max(xs))}

def material(batch, n_values):
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from tfpd_exploration.src.m1_optimized_v2.source_dev import _model
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank
    report=json.loads(POST.read_text()); row=report["selected"]
    state=Path(row["plain_ema_model_state"])
    if sha(state)!=row["plain_ema_model_state_sha256"]: raise RuntimeError("selected EMA state drift")
    cache_path=plan.RESULT_ROOT/"m1_optimized_v2_source_runtime_cache.npz"
    receipt=json.loads(cache_path.with_suffix(".receipt.json").read_text())
    if sha(cache_path)!=receipt["npz_sha256"]: raise RuntimeError("runtime cache drift")
    names=[*plan.SOURCE_SESSIONS,plan.SOURCE_SESSIONS[0]][:batch]
    cache=np.load(cache_path,allow_pickle=False)
    model=_model("current_query").eval(); model.load_state_dict(torch.load(state,map_location="cpu",weights_only=False),strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    e0=torch.stack([torch.from_numpy(cache[f"bank_e0/{n}"]).float() for n in names])
    tt=torch.stack([torch.from_numpy(cache[f"bank_t/{n}"]).float() for n in names])
    mask=torch.stack([torch.from_numpy(cache[f"bank_unit_mask/{n}"]).bool() for n in names])
    bank=BankBatch(e0,tt,mask,tuple(names),tuple(tuple(range(64)) for _ in names))
    refbanks=[M1Bank(e0[i],tt[i],mask[i]) for i in range(batch)]
    values=np.stack([cache[f"raw_neural/{n}"][99:99+n_values] for n in names],axis=1).astype(np.float32,copy=False)
    cache.close()
    return model,bank,refbanks,values,row,receipt,names

def main(a):
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    t=time.perf_counter_ns(); model,bank,refbanks,values,row,receipt,names=material(a.batch,1+a.warmup+a.calls); load_ms=(time.perf_counter_ns()-t)/1e6
    if a.mode=="v3":
        t=time.perf_counter_ns(); engine=HeterogeneousCurrentQueryStream(model,bank); construct_ms=(time.perf_counter_ns()-t)/1e6; call=engine.predict
        persistent={"all_runtime_owned_bytes":engine.state_bytes,"rolling_bytes":engine.rolling_state_bytes,"static_derived_bytes":engine.static_cache_bytes}
    elif a.mode=="reference_cached":
        from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import CurrentQueryStream
        t=time.perf_counter_ns(); engine=[CurrentQueryStream(model,x,task="m1",session_id=n,unit_ids=tuple(range(64))) for x,n in zip(refbanks,names,strict=True)]; construct_ms=(time.perf_counter_ns()-t)/1e6
        def call(value): return np.concatenate([x.predict(value[i:i+1]) for i,x in enumerate(engine)],axis=0)
        persistent={"sum_stream_state_bytes":sum(x.state_bytes for x in engine)}
    else:
        from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import CurrentQueryStream
        from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1Bank
        t=time.perf_counter_ns(); engine=CurrentQueryStream(model,M1Bank(bank.E0,bank.T,bank.unit_mask),task="m1",session_id=tuple(names),unit_ids=tuple(range(64)),batch_size=a.batch); construct_ms=(time.perf_counter_ns()-t)/1e6
        call=engine.predict; persistent={"stream_state_bytes":engine.state_bytes}
    t=time.perf_counter_ns(); first=call(values[0]); first_ms=(time.perf_counter_ns()-t)/1e6
    if first.shape!=(a.batch,16) or first.dtype!=np.float32 or not np.isfinite(first).all(): raise RuntimeError("public API contract")
    for x in values[1:1+a.warmup]: call(x)
    samples=[]; peak=rss()
    for x in values[1+a.warmup:1+a.warmup+a.calls]:
        t=time.perf_counter_ns(); y=call(x); samples.append((time.perf_counter_ns()-t)/1e6); peak=max(peak,rss())
        if y.shape!=(a.batch,16) or y.dtype!=np.float32 or not np.isfinite(y).all(): raise RuntimeError("public API drift")
    out={"schema":"m1_runtime_v3_public_micro_v1","mode":a.mode,"threads":a.threads,"batch":a.batch,"calls":a.calls,"warmup":a.warmup,"source_rows":names,"fourth_lane":"repeat ses-20120926 when B4","inputs":"cached source raw_neural with exactly 99 startup bins stripped once; no NWB access","model_load_selected_ema_ms":load_ms,"construct_reset_ms":construct_ms,"first_public_return_ms":first_ms,"public_predict_whole_batch":qs(samples),"persistent":persistent,"peak_process_hwm_bytes":peak,"selected_ema_state_sha256":row["plain_ema_model_state_sha256"],"runtime_cache_sha256":receipt["npz_sha256"],"runtime_code_sha256":sha(Path(__file__).with_name("runtime.py")),"affinity":sorted(os.sched_getaffinity(0)),"not_container_or_official_latency":True}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,sort_keys=True))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("reference_cached","reference_heterobatched","v3"),required=True);p.add_argument("--batch",type=int,choices=(1,4),required=True);p.add_argument("--threads",type=int,choices=(1,2),required=True);p.add_argument("--calls",type=int,default=256);p.add_argument("--warmup",type=int,default=128);p.add_argument("--output",type=Path,required=True);main(p.parse_args())
