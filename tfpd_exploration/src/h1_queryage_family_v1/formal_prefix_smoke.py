"""Disposable per-arm resource smoke for the prospective H1 QueryAge+p=.5 run.

There is intentionally no CLI.  A root supervisor supplies the cross-process
barrier and is the only component allowed to launch this worker on a GPU.
"""
from __future__ import annotations
import hashlib,json,os,random,tempfile,time
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import torch
from . import formal_prefix_train as train

SMOKE_UPDATES,WARM,STEADY,HARD_SECONDS,HARD_MEMORY=20,4,16,900,22<<30
SCHEMA="h1_queryage_formal_prefix_smoke_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _json(p):return json.loads(Path(p).read_text())
def _atomic(p,v):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",dir=Path(p).parent,delete=False) as f:t=Path(f.name);json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def bindings(output,*,capacity_receipt,capacity_checkpoint,physical_gpu,threads=1):
 b=train.collect_bindings(Path(output),capacity_receipt=Path(capacity_receipt),capacity_checkpoint=Path(capacity_checkpoint))
 src=train.SRC
 proxy_files=(src/'family_runtime_v1/diagnose_h1_selected_source208.py',src/'family_runtime_v1/diagnose_h1_endpoint_raw.py',src/'h1_optimized_v2/capacity_probe.py',train.ROOT/'results/decoder_validation_v2/20260905_190000/h1/capacity_probe_208_source_v2/frozen_ids.json')
 return {"schema":SCHEMA,"formal":b,"output":str(Path(output)),"physical_gpu":physical_gpu,"threads":threads,"updates":SMOKE_UPDATES,"warm":WARM,"steady":STEADY,"code":{"smoke":sha(Path(__file__)),"trainer":sha(Path(train.__file__))},"source208_proxy_files":{str(p):sha(p) for p in proxy_files}}
def _authorize(binding,authorization,authorization_sha):
 if not Path(authorization).is_absolute() or not Path(authorization).is_file():raise RuntimeError("absolute external smoke authorization required")
 if sha(authorization)!=authorization_sha or _json(authorization).get("status")!="ROOT_REVIEW_GO" or _json(authorization).get("bindings")!=binding:raise RuntimeError("smoke external authorization drift")
def _guard(started,device):
 if device.type=="cuda":
  torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
  if peak>HARD_MEMORY:raise RuntimeError("smoke 22-GiB memory bound")
 if time.monotonic()-started>HARD_SECONDS:raise TimeoutError("smoke 900-second bound")
def _forecast(timing,per_endpoint_seconds,reload_seconds):
 if len(timing)!=SMOKE_UPDATES or len(timing[WARM:])!=STEADY or not np.isfinite([*timing,per_endpoint_seconds,reload_seconds]).all() or min([*timing,per_endpoint_seconds,reload_seconds])<=0:raise RuntimeError("finite positive exact smoke timing/proxy/reload required")
 p95=float(np.percentile(np.asarray(timing[WARM:],float),95));train_wall=p95*12*731*1.5
 endpoint=per_endpoint_seconds*1.5
 return {"per_arm_smoke_p95_seconds":p95,"source208_seconds_per_endpoint":per_endpoint_seconds,"concurrent_train_seconds":train_wall,"selection_proxy_seconds":endpoint*12*2908,"complete_reporting_seconds":endpoint*2*20325,"reload_seconds":reload_seconds*12,"setup_margin_seconds":1800.,"conservative_total_seconds":train_wall+endpoint*(12*2908+2*20325)+reload_seconds*12+1800.}
def _ready_payload(*,arm,physical_gpu,shared,ids,parity,gate):return {"schema":SCHEMA,"arm":arm,"physical_gpu":physical_gpu,"shared_init_sha256":shared,"identities":ids,"g0_parity":parity,"route_gate_gradient_l1":gate}
def verify_peer(local,peer):
 for key in ("schema","shared_init_sha256","identities"):
  if local.get(key)!=peer.get(key):raise RuntimeError("split-arm smoke matching barrier drift")
 if local.get("arm")==peer.get("arm") or local.get("physical_gpu")==peer.get("physical_gpu") or local.get("g0_parity",{}).get("max_abs_diff")!=0. or peer.get("g0_parity",{}).get("max_abs_diff")!=0. or not np.isfinite([float(local.get("route_gate_gradient_l1",0)),float(peer.get("route_gate_gradient_l1",0))]).all() or float(local.get("route_gate_gradient_l1",0))<=0 or float(peer.get("route_gate_gradient_l1",0))<=0:raise RuntimeError("smoke paired gate evidence drift")
def _wait_peer(path,timeout_seconds=300.,guard=lambda:None):
 deadline=time.monotonic()+timeout_seconds;path=Path(path)
 while not path.is_file():
  guard()
  if time.monotonic()>deadline:raise TimeoutError("paired smoke peer-ready timeout")
  time.sleep(.05)
 return _json(path)
def _source208_proxy(model,ema,cache,device,guard,*,fixed_ids=None,evaluator=None):
 """Actual source-train-only EMA forward timing, normalized per fixed endpoint."""
 from tfpd_exploration.src.family_runtime_v1.diagnose_h1_selected_source208 import _require_ids,evaluate_source208
 from tfpd_exploration.src.family_runtime_v1.diagnose_h1_endpoint_raw import GuardedModel
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
 # `score_with_ema` is deliberately the same temporary-shadow substitution
 # used by the formal scorer.  This measures EMA inference, never the RAW
 # model merely because the latter happens to share the same topology.
 fixed=_require_ids() if fixed_ids is None else fixed_ids;guard();begun=time.perf_counter()
 def bank_factory(row,d):guard();return H1Bank(*[row["bank"][k].to(d) for k in ("E0","T","unit_mask")])
 def forward(ema_model):
  class MicrobatchModel:
   def eval(self):ema_model.eval();return self
   def forward_last(self,x,bank):
    parts=[]
    for first in range(0,len(x),train.MICRO):
     guard();parts.append(ema_model.forward_last(x[first:first+train.MICRO],bank));guard()
    return torch.cat(parts,dim=0)
  # The existing fixed208 evaluator supplies B16 per session; use the formal
  # scorer's actual micro8 boundary for a correctly dimensioned cost forecast.
  guarded=GuardedModel(MicrobatchModel(),lambda *_:guard())
  if evaluator is None:evaluate_source208(guarded,cache,fixed,device,bank_factory)
  else:evaluator(guarded,cache,fixed,device,bank_factory)
 ema.score_with_ema(model,forward);guard()
 return (time.perf_counter()-begun)/208.
def run_smoke(*,arm,output,authorization,authorization_sha,peer_ready,start_marker,capacity_receipt,capacity_checkpoint,physical_gpu,allow_cpu_fixture=False,source_loader=None,model_factory=None,source208_evaluator=None,source208_ids=None):
 started=time.monotonic()
 if arm not in train.ARMS or physical_gpu!=train.ARM_DEVICE[arm]:raise RuntimeError("smoke arm/device drift")
 if os.environ.get("H1_QUERYAGE_FORMAL_PREFIX_GO")!="1" or (not allow_cpu_fixture and os.environ.get("CUDA_VISIBLE_DEVICES")!=str(physical_gpu)):raise RuntimeError("formal smoke GO/device environment required")
 supervisor=Path(output);output=supervisor/arm;binding=bindings(output,capacity_receipt=capacity_receipt,capacity_checkpoint=capacity_checkpoint,physical_gpu=physical_gpu)
 if output.exists():raise FileExistsError(output)
 _authorize(binding,Path(authorization),authorization_sha)
 if not allow_cpu_fixture and (not torch.cuda.is_available() or torch.cuda.current_device()!=0):raise RuntimeError("smoke requires visible cuda:0")
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:
  if torch.get_num_interop_threads()!=1:raise
 from tfpd_exploration.src.h1_optimized_v4.paired_train import batches,collate
 from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
 from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
 from tfpd_exploration.src.h1_family_v1.model import initialization_receipt,route_gate_gradient_l1,zero_gate_parity
 from .model import make_queryage_localbalanced_pair
 if any(value is not None for value in (model_factory,source208_evaluator,source208_ids)) and not allow_cpu_fixture:raise RuntimeError("smoke fixtures are test-only")
 device=torch.device("cpu" if allow_cpu_fixture else "cuda:0")
 if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
 _guard(started,device)
 cache=train.load_readonly_cache(source_loader=source_loader,allow_cpu_fixture=allow_cpu_fixture);_guard(started,device)
 ordered=batches(cache,1);ids=train._all_epoch_identities(cache);_guard(started,device);session,starts=ordered[0];row=cache["train"][session];x,y=collate(row,starts,torch.device("cpu"));bank=H1Bank(row["bank"]["E0"],row["bank"]["T"],row["bank"]["unit_mask"])
 # Probe is deliberately destroyed; no probe state enters smoke training.
 pair_factory=make_queryage_localbalanced_pair if model_factory is None else model_factory
 flat,route=pair_factory(seed=train.SEED);shared=train.state_digest(flat.state_dict());probe_all,_,keep_all=train.prefix_and_keep(x,epoch=1,batch_index=0,bank_mask=bank.unit_mask,device=device);probe,keep=probe_all[:train.MICRO],keep_all[:train.MICRO];bank_probe=H1Bank(*[row["bank"][k].to(device) for k in ("E0","T","unit_mask")]);_guard(started,device);parity=zero_gate_parity(flat.to(device),route.to(device),probe,bank_probe);_guard(started,device);route.zero_grad(set_to_none=True);_guard(started,device);torch.nn.functional.mse_loss(route.forward_last(probe,bank_probe,dropout_keep=keep),y[:train.MICRO].to(device)).backward();_guard(started,device);gate=route_gate_gradient_l1(route)
 if parity["max_abs_diff"]!=0 or gate<=0:raise RuntimeError("fresh paired probe failed")
 del flat,route,probe,keep,probe_all,keep_all
 torch.manual_seed(train.SEED);np.random.seed(train.SEED);random.seed(train.SEED);flat,route=pair_factory(seed=train.SEED)
 if train.state_digest(flat.state_dict())!=shared:raise RuntimeError("fresh post-probe state differs from seed42 initialization")
 model=(flat if arm=="flat" else route).to(device)
 if arm=="flat": del route
 else: del flat
 ready=_ready_payload(arm=arm,physical_gpu=physical_gpu,shared=shared,ids=ids,parity=parity,gate=gate);_atomic(output/"ready.json",ready);ready_sha=sha(output/"ready.json");_atomic(output/"input_authority.json",{"authority":binding,"external_authorization_sha256":authorization_sha});own_input_sha=sha(output/"input_authority.json");verify_peer(ready,_wait_peer(peer_ready,guard=lambda:_guard(started,device)));peer_sha=sha(peer_ready)
 _guard(started,device)
 deadline=time.monotonic()+300.
 while not Path(start_marker).is_file():
  _guard(started,device)
  if time.monotonic()>deadline:raise TimeoutError("supervisor start timeout")
  time.sleep(.05)
 opt=torch.optim.AdamW(groups(model),lr=train.LR,weight_decay=.01);ema=DecoderEMA(model,decay=train.EMA);times=[];losses=[]
 for i,(session,starts) in enumerate(ordered[:SMOKE_UPDATES]):
  row=cache["train"][session];xb,yb=collate(row,starts,device);b=H1Bank(*[row["bank"][k].to(device) for k in ("E0","T","unit_mask")]);lr=train.warmup_lr(epoch=1,global_step=i+1)
  for g in opt.param_groups:g["lr"]=lr
  _guard(started,device);begun=time.perf_counter();loss,_=train._train_update(model,opt,ema,xb,yb,b,epoch=1,batch_index=i,device=device,guard=lambda:_guard(started,device));_guard(started,device);times.append(time.perf_counter()-begun);losses.append(loss)
 if len(times)!=SMOKE_UPDATES or not np.isfinite(losses).all():raise RuntimeError("smoke paired update/timing drift")
 temp=output/"known_disposable_smoke_checkpoint.pt";payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"ema":ema.checkpoint_state(),"rng":{"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":__import__("random").getstate(),"cuda":torch.cuda.get_rng_state_all()}}
 try:
  begun=time.perf_counter();train.atomic_torch_save(payload,temp);_guard(started,device);loaded=torch.load(temp,map_location="cpu",weights_only=False);_guard(started,device)
  if train.state_digest(loaded["model"])!=train.state_digest(model.state_dict()) or not train.same(loaded["optimizer"],opt.state_dict()) or not train.same(loaded["ema"],ema.checkpoint_state()) or not train.same(loaded["rng"],payload["rng"]):raise RuntimeError("smoke checkpoint reload drift")
  model.load_state_dict(loaded["model"],strict=True);opt.load_state_dict(loaded["optimizer"]);ema.load_checkpoint_state(loaded["ema"]);ema.shadow={k:v.to(device) for k,v in ema.shadow.items()};torch.set_rng_state(loaded["rng"]["torch"]);np.random.set_state(loaded["rng"]["numpy"]);random.setstate(loaded["rng"]["python"]);torch.cuda.set_rng_state_all(loaded["rng"]["cuda"]);_guard(started,device);reload_seconds=time.perf_counter()-begun
 finally:
  temp.unlink(missing_ok=True)
 if temp.exists():raise RuntimeError("known disposable checkpoint survived")
 per_endpoint=_source208_proxy(model,ema,cache,device,lambda:_guard(started,device),fixed_ids=source208_ids,evaluator=source208_evaluator);forecast=_forecast(times,per_endpoint,reload_seconds)
 _guard(started,device);post=bindings(output,capacity_receipt=capacity_receipt,capacity_checkpoint=capacity_checkpoint,physical_gpu=physical_gpu)
 if post!=binding:raise RuntimeError("fresh smoke code/input authority drift")
 _authorize(binding,Path(authorization),authorization_sha)
 if sha(output/"ready.json")!=ready_sha or sha(peer_ready)!=peer_sha or sha(output/"input_authority.json")!=own_input_sha:raise RuntimeError("smoke owned/peer authority mutation")
 result={"schema":SCHEMA,"status":"PASS_DISPOSABLE_SOURCE_ONLY_SMOKE","authority":binding,"arm":arm,"timing":times,"losses":losses,"forecast":forecast,"proxy":"actual guarded source208 EMA timing only; no minival/complete scoring","parameter_updates":SMOKE_UPDATES,"checkpoint_retained":False,"capacity_state_used_for_warmstart":False,"whole_cache_may_contain_minival_bytes":True,"peak_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,"elapsed_seconds":time.monotonic()-started}
 result.update({"authority_post":post,"ready_sha256":ready_sha,"peer_ready_sha256":peer_sha,"input_authority_sha256":own_input_sha,"external_authorization_sha256":authorization_sha,"reload_seconds_measured":reload_seconds,"source208_seconds_per_endpoint":per_endpoint})
 _guard(started,device);_atomic(output/"receipt.json",result);return result
