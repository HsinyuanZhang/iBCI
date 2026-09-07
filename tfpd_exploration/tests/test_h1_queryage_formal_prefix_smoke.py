from __future__ import annotations
import json
import numpy as np,pytest,torch
from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_smoke as s
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA

def test_forecast_uses_only_steady_p95_and_declared_complete_budget():
 f=s._forecast([9.,8.,7.,6.]+[2.]*16,.01,2.)
 assert f["per_arm_smoke_p95_seconds"]==2. and f["conservative_total_seconds"]>f["concurrent_train_seconds"]
def test_matching_ready_requires_shared_schedule_and_both_gate_evidence():
 base={"schema":s.SCHEMA,"arm":"flat","physical_gpu":0,"shared_init_sha256":"x","identities":{"1":{"a":"b"}},"g0_parity":{"max_abs_diff":0.},"route_gate_gradient_l1":1.}
 peer={**base,"arm":"route","physical_gpu":1}
 s.verify_peer(base,peer)
 bad=dict(base);bad["identities"]={}
 with pytest.raises(RuntimeError,match="matching barrier"):s.verify_peer(base,bad)
def test_external_authorization_is_hash_bound(tmp_path):
 p=tmp_path/"a.json";p.write_text('{"status":"ROOT_REVIEW_GO","bindings":{"x":1}}')
 with pytest.raises(RuntimeError,match="authorization drift"):s._authorize({"x":2},p,s.sha(p))

def test_source208_proxy_uses_ema_shadow_not_raw_model():
 class Tiny(torch.nn.Module):
  def __init__(self): super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
  def forward_last(self,x,bank): return x*self.weight
 model=Tiny();ema=DecoderEMA(model,decay=.9);ema.update_after_step(model)
 with torch.no_grad():model.weight.fill_(9.)
 seen=[];calls=[]
 def evaluator(proxy,cache,fixed,device,bank_factory):
  seen.append(float(proxy.forward_last(torch.ones(1),None)))
 def guard():calls.append(1)
 value=s._source208_proxy(model,ema,{},torch.device("cpu"),guard,fixed_ids={},evaluator=evaluator)
 assert value>=0 and seen==[2.] and float(model.weight)==9. and len(calls)>=3

def test_reduced_cpu_run_smoke_exercises_full_disposable_lifecycle(tmp_path,monkeypatch):
 """CPU toy fixture still performs the production count: 20 real updates."""
 import tfpd_exploration.src.h1_family_v1.model as family_model
 import tfpd_exploration.src.h1_optimized_v2.cache as cache_module
 import tfpd_exploration.src.h1_optimized_v4.paired_train as sampler
 import tfpd_exploration.src.h1_queryage_family_v1.model as queryage_model
 class Tiny(torch.nn.Module):
  def __init__(self,seed):
   super().__init__();torch.manual_seed(seed);self.readout=torch.nn.Linear(176,7)
  def forward_last(self,x,bank,dropout_keep=None): return self.readout(x.mean(dim=1))
 def pair(*,seed): return Tiny(seed),Tiny(seed)
 bank={"E0":torch.zeros(176,1),"T":torch.zeros(176,1),"unit_mask":torch.ones(176,dtype=torch.bool)}
 cache={"train":{f"s{i}":{"bank":bank} for i in range(13)},"minival":{}}
 starts=np.arange(8,dtype=np.int64)
 monkeypatch.setattr(s,"SMOKE_UPDATES",20);monkeypatch.setattr(s,"WARM",4);monkeypatch.setattr(s,"STEADY",16)
 monkeypatch.setenv("H1_QUERYAGE_FORMAL_PREFIX_GO","1")
 monkeypatch.setattr(queryage_model,"make_queryage_localbalanced_pair",pair)
 monkeypatch.setattr(family_model,"zero_gate_parity",lambda *a,**k:{"max_abs_diff":0.})
 monkeypatch.setattr(family_model,"route_gate_gradient_l1",lambda *a,**k:1.)
 monkeypatch.setattr(cache_module,"validate_authority",lambda *a,**k:None)
 monkeypatch.setattr(sampler,"batches",lambda *a,**k:[("s0",starts)]*20)
 monkeypatch.setattr(sampler,"collate",lambda row,starts,device:(torch.ones((len(starts),700,176),device=device),torch.zeros((len(starts),7),device=device)))
 monkeypatch.setattr(s.train,"_all_epoch_identities",lambda *a,**k:{"fixture":"toy-20-updates"})
 root=tmp_path/"smoke";capacity_receipt=tmp_path/"capacity.json";capacity_checkpoint=tmp_path/"capacity.pt"
 capacity_receipt.write_text("{}\n");torch.save({},capacity_checkpoint)
 binding=s.bindings(root/"flat",capacity_receipt=capacity_receipt,capacity_checkpoint=capacity_checkpoint,physical_gpu=0)
 authorization=tmp_path/"authorization.json";authorization.write_text(json.dumps({"status":"ROOT_REVIEW_GO","bindings":binding}))
 peer=root/"route"/"ready.json";peer.parent.mkdir(parents=True)
 s._atomic(peer,s._ready_payload(arm="route",physical_gpu=1,shared=s.train.state_digest(Tiny(42).state_dict()),ids={"fixture":"toy-20-updates"},parity={"max_abs_diff":0.},gate=1.))
 start=tmp_path/"START";start.write_text("released\n")
 result=s.run_smoke(arm="flat",output=root,authorization=authorization,authorization_sha=s.sha(authorization),peer_ready=peer,start_marker=start,capacity_receipt=capacity_receipt,capacity_checkpoint=capacity_checkpoint,physical_gpu=0,allow_cpu_fixture=True,source_loader=lambda:(cache,{}),model_factory=pair,source208_ids={},source208_evaluator=lambda proxy,*a:proxy.forward_last(torch.ones((1,700,176)),None))
 receipt=json.loads((root/"flat"/"receipt.json").read_text())
 assert result==receipt and receipt["schema"]==s.SCHEMA and receipt["status"]=="PASS_DISPOSABLE_SOURCE_ONLY_SMOKE"
 assert receipt["arm"]=="flat" and receipt["authority"]==binding and receipt["authority_post"]==binding
 assert receipt["external_authorization_sha256"]==s.sha(authorization)
 assert receipt["proxy"]=="actual guarded source208 EMA timing only; no minival/complete scoring"
 assert receipt["parameter_updates"]==20 and receipt["checkpoint_retained"] is False
 assert receipt["capacity_state_used_for_warmstart"] is False and len(receipt["timing"])==20
 assert not (root/"flat"/"known_disposable_smoke_checkpoint.pt").exists()
