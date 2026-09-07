"""Full fixture integration for the authorized cached QueryAge B7 benchmark."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pytest, torch
from tfpd_exploration.src.family_runtime_v1 import compare_m2_queryage_cached_spint as c
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class _Original:
 def __init__(self,*a,**k):
  self.observation_buffer=np.zeros((50,7,96),np.float32);self.device=torch.device("cpu");self.smooth_observations=False;self.local_identities=[None]*7;self.behavior_scaling_factor=1.
 def reset(self,*a): pass
 def predict(self,x): self.observation_buffer[:-1]=self.observation_buffer[1:];self.observation_buffer[-1]=x;return np.zeros((7,2),np.float32).copy()
 def _decode_with_identity(self,x,identity): return torch.zeros((1,50,2))
class _Wrapper: T4CachedIdentityDecoder=_Original

def _fixture(tmp_path, monkeypatch):
 root=(tmp_path/"run").resolve();final=(tmp_path/"final").resolve();root.mkdir();final.mkdir();(root/"protocol_preconstruction.json").write_text("{}")
 cache=root/"cache"/"source_train";authority={}
 for number,session in enumerate(c.plan.HELDIN_SESSIONS):
  d=cache/session;d.mkdir(parents=True);raw=np.zeros((210,96),np.float32);raw[49:]=np.random.default_rng(number).normal(0,.03,(161,96)).astype(np.float32)
  np.save(d/"X_store.npy",raw);np.save(d/"target_store.npy",np.zeros((2,2),np.float32));np.save(d/"eligible_starts.npy",np.array([0,1],np.int64));np.save(d/"T.npy",np.zeros((96,4),np.float32));torch.save({"E0":torch.zeros((96,50),dtype=torch.float32)},d/"e0_u.pt")
  for name in ("mapping.json","provenance.json","extra.json"): (d/name).write_text("{}")
  np.save(d/"calib_activity.npy",np.zeros(96,np.float32))
  for p in d.iterdir(): authority[str(p)]=_sha(p)
 monkeypatch.setattr(c.plan,"active_run_root",lambda:root)
 selected={}
 for arm,m in zip(("FLAT","ROUTE"),make_paired_queryage_decoders(42),strict=True):
  if arm=="ROUTE":m.frontend.attn.routing.g.data.fill_(.2)
  p=tmp_path/(arm+".pt");torch.save(m.state_dict(),p);selected[arm]={"key":arm,"path":str(p),"sha256":_sha(p)}
 audit={"selected_primary_ema_epoch":{"FLAT":1,"ROUTE":1}}
 receipt={"authority_pre_model":{"fixture":"minival"}}
 monkeypatch.setattr(c.native,"_selected_exports",lambda *a:(audit,receipt,selected))
 monkeypatch.setattr(c.finalizer,"_current_trainer_authority",lambda *a:authority)
 # Stable native proof boundary; all proof JSON/archive bytes are real fixture files.
 native_authority={"run_root":str(root),"finalizer_root":str(final),"finalizer_receipt_sha256":"a"*64,"output":str(tmp_path/"native"),"selected":selected}
 monkeypatch.setattr(c.complete,"_native_authority",lambda *a: ({},native_authority))
 proofs={};code=c.complete.runtime_code()
 for kind in ("uncached","cached"):
  pdir=tmp_path/("proof_"+kind);pdir.mkdir();side=pdir/"authority_pre.json";pre={"runtime_code_sha256":code,"native_authority":native_authority,"native_receipt_sha256":"b"*64};side.write_text(json.dumps({"authority":pre}))
  results={}
  for arm in ("FLAT","ROUTE"):
   a=pdir/(arm+".npz");np.savez_compressed(a,prediction=np.zeros((1,2)));results[arm]={"archive_path":str(a),"archive_sha256":_sha(a)}
  body={"status":"PASS_COMPLETE_QUERYAGE_STREAM_EQUIVALENCE_ONLY","runtime_kind":kind,"authority_pre":pre,"authority_post":pre,"authority_pre_sha256":_sha(side),"results":results};rp=pdir/"receipt.json";rp.write_text(json.dumps(body));proofs[kind]=(pdir,_sha(rp))
 # wrapper/payload provenance is a real fixture closure rather than image bytes.
 frozen=tmp_path/"frozen.py";frozen.write_text("fixture\n");payload=tmp_path/"decoder.pkl";payload.write_bytes(b"fixture-payload")
 monkeypatch.setattr(c,"EXPECTED",{frozen:_sha(frozen),payload:_sha(payload)});monkeypatch.setattr(c,"AS_SHIPPED",payload);monkeypatch.setattr(c,"WRAPPER",frozen)
 final_receipt=final/"receipt.json";final_receipt.write_text("{}")
 return root,final,selected,proofs

def test_full_collect_authorize_run_persisted_receipt_actual_cached_queryage(tmp_path,monkeypatch):
 root,final,selected,proofs=_fixture(tmp_path,monkeypatch);out=(tmp_path/"out.json").resolve()
 # collect reads seven real fixture folders and validates exact native proof metadata.
 monkeypatch.setattr(c,"sha",c.sha)
 # Finalizer receipt hash is externally fixed only at this fixture boundary.
 monkeypatch.setattr(c,"_json",lambda p: {"audit": {"selected_primary_ema_epoch":{"FLAT":1,"ROUTE":1}}} if Path(p)==final/"selection_freeze.json" else json.loads(Path(p).read_text()))
 frsha=_sha(final/"receipt.json")
 binding=c.collect_bindings(root,final,frsha,{k:v[0] for k,v in proofs.items()},{k:v[1] for k,v in proofs.items()},out,calls=32,warmup=128,threads=1)
 auth=tmp_path/"auth.json";auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":binding}))
 monkeypatch.setenv(c.GO,"1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(c.torch.cuda,"is_available",lambda:False)
 result=c.run(binding,authorization=auth,authorization_sha=_sha(auth),wrapper=_Wrapper,config=None,allow_cpu_fixture=True)
 assert result["status"]=="PASS_PUBLIC_STREAM_TIMING_ONLY" and out.is_file()
 assert result["authority_pre_sha256"]==_sha(Path(result["authority_pre_path"])) and set(result["steady_public_call_ms"])=={"ORIGINAL","FLAT","ROUTE"}
 assert all(v<=1e-5 for v in result["candidate_oracle_max_abs_error"].values())

def test_gate_rejects_bad_auth_before_model(tmp_path,monkeypatch):
 binding={"calls":32,"warmup":128,"threads":1,"interop_threads":1};auth=tmp_path/"a.json";auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":{}}))
 monkeypatch.setenv(c.GO,"1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(c.torch.cuda,"is_available",lambda:False)
 with pytest.raises(RuntimeError,match="authorization drift"): c.run(binding,authorization=auth,authorization_sha=_sha(auth),wrapper=_Wrapper,allow_cpu_fixture=True)

def test_final_authorization_mutation_refuses_completion(tmp_path,monkeypatch):
 root,final,selected,proofs=_fixture(tmp_path,monkeypatch);out=(tmp_path/"out.json").resolve()
 monkeypatch.setattr(c,"_json",lambda p: {"audit":{"selected_primary_ema_epoch":{"FLAT":1,"ROUTE":1}}} if Path(p)==final/"selection_freeze.json" else json.loads(Path(p).read_text()))
 binding=c.collect_bindings(root,final,_sha(final/"receipt.json"),{k:v[0] for k,v in proofs.items()},{k:v[1] for k,v in proofs.items()},out,calls=32,warmup=128,threads=1)
 auth=tmp_path/"auth.json";auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":binding}))
 class Mutating(_Original):
  calls=0
  def predict(self,x):
   value=super().predict(x);type(self).calls+=1
   if type(self).calls==161: auth.write_text('{"status":"MUTATED"}')
   return value
 class MutatingWrapper: T4CachedIdentityDecoder=Mutating
 monkeypatch.setenv(c.GO,"1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(c.torch.cuda,"is_available",lambda:False)
 with pytest.raises(RuntimeError,match="authorization changed"):
  c.run(binding,authorization=auth,authorization_sha=_sha(auth),wrapper=MutatingWrapper,config=None,allow_cpu_fixture=True)
 assert not out.exists()
