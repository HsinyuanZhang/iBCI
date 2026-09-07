"""Synthetic CPU public-boundary checks for the prospective QueryAge benchmark."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pytest,torch
from tfpd_exploration.src.family_runtime_v1 import compare_m2_queryage_cached_spint as c
from tfpd_exploration.src.family_runtime_v1.m2_family_causal import M2RuntimeBank
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders

class Original:
 def __init__(self,*a,**k):self.raw=np.zeros((7,50,96),np.float32);self.observation_buffer=np.zeros((50,7,96),np.float32);self.device=torch.device('cpu');self.smooth_observations=False;self.local_identities=[None]*7;self.behavior_scaling_factor=1.
 def reset(self,*a):pass
 def predict(self,x):self.raw[:,:-1]=self.raw[:,1:];self.raw[:,-1]=x;self.observation_buffer[:-1]=self.observation_buffer[1:];self.observation_buffer[-1]=x;return np.zeros((7,2),np.float32).copy()
 def _decode_with_identity(self,neural,identity):return torch.zeros((1,50,2),dtype=torch.float32)
class Wrapper:T4CachedIdentityDecoder=Original
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_three_engine_rotation_real_cached_queryage_public_boundary(tmp_path,monkeypatch):
 torch.set_num_threads(1); paths={}
 for arm,m in zip(("FLAT","ROUTE"),make_paired_queryage_decoders(42),strict=True):
  p=tmp_path/(arm+'.pt');torch.save(m.state_dict(),p);paths[arm]={"key":arm,"path":str(p),"sha256":_sha(p)}
 g=torch.Generator().manual_seed(7);bank=M2RuntimeBank(torch.randn(7,96,50,generator=g)*.1,torch.randn(7,96,4,generator=g)*.1,torch.ones(7,96,dtype=torch.bool));values=np.random.default_rng(1).normal(0,.1,(161,7,96)).astype(np.float32)
 binding={"selected":paths,"calls":32,"warmup":128,"threads":1,"interop_threads":1,"image":c.IMAGE,"schema":c.SCHEMA}
 auth=tmp_path/'auth.json';auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":binding}));monkeypatch.setenv(c.GO,"1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(c.torch.cuda,"is_available",lambda:False)
 out=c.run(binding,authorization=auth,authorization_sha=_sha(auth),wrapper=Wrapper,config=None,source=([str(i) for i in range(7)],bank,values,{"x":"y"}),allow_cpu_fixture=True)
 assert out["rotating_three_engine_order"] and set(out["steady_public_call_ms"])=={"ORIGINAL","FLAT","ROUTE"}
 assert all(v<=1e-5 for v in out["candidate_oracle_max_abs_error"].values())
def test_collect_refuses_bad_mode_without_artifacts(tmp_path):
 with pytest.raises(RuntimeError,match="32/2048"):
  c.collect_bindings(tmp_path,tmp_path,"a"*64,{}, {},tmp_path/'o',calls=31,warmup=128,threads=1)

def test_run_refuses_authorization_before_any_model_or_timer(tmp_path,monkeypatch):
 binding={"schema":c.SCHEMA,"image":c.IMAGE,"calls":32,"warmup":128,"threads":1,"interop_threads":1}
 auth=tmp_path/'auth.json';auth.write_text(json.dumps({"status":"ROOT_REVIEW_GO","authority":{"different":True}}))
 monkeypatch.setenv(c.GO,"1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setattr(c.torch.cuda,"is_available",lambda:False)
 with pytest.raises(RuntimeError,match="authorization drift"):
  c.run(binding,authorization=auth,authorization_sha=_sha(auth),wrapper=Wrapper,source=None,allow_cpu_fixture=True)

def test_source_train_nine_file_absolute_trainer_closure_before_load(tmp_path,monkeypatch):
 root=tmp_path/'active';closure={}
 for i,session in enumerate(c.plan.HELDIN_SESSIONS):
  folder=root/'cache'/'source_train'/session;folder.mkdir(parents=True)
  raw=np.full((220,96),i,dtype=np.float32);raw[:49]=0;np.save(folder/'X_store.npy',raw);np.save(folder/'target_store.npy',np.zeros((1,2),np.float32));np.save(folder/'eligible_starts.npy',np.zeros(1,np.int64));np.save(folder/'T.npy',np.zeros((96,4),np.float32));torch.save({'E0':torch.zeros(96,50,dtype=torch.float32)},folder/'e0_u.pt')
  for name in ('mapping.json','provenance.json','extra.json'):(folder/name).write_text('{}')
  np.save(folder/'calib_activity.npy',np.zeros(96,np.float32))
  for p in folder.iterdir():closure[str(p)]=_sha(p)
 monkeypatch.setattr(c.plan,'active_run_root',lambda:root)
 tags,bank,values,actual=c._source(161,closure)
 assert len(tags)==7 and bank.E0.shape==(7,96,50) and values.shape==(161,7,96) and actual[c.plan.HELDIN_SESSIONS[0]]['X_store.npy']==closure[str(root/'cache'/'source_train'/c.plan.HELDIN_SESSIONS[0]/'X_store.npy')]
 (root/'cache'/'source_train'/c.plan.HELDIN_SESSIONS[0]/'extra.json').write_text('{"drift":true}')
 with pytest.raises(RuntimeError,match='source-train authority'):_=c._source(161,closure)

def test_completed_proof_metadata_rechecks_runtime_and_native_authority(tmp_path,monkeypatch):
 root=tmp_path/'proof';root.mkdir();code={'cached.py':'a'*64};native={'run_root':'/run','finalizer_root':'/fin','finalizer_receipt_sha256':'b'*64,'output':'/native'}
 archives={}
 for arm in ('FLAT','ROUTE'):
  p=root/(arm+'.npz');np.savez(p,x=np.zeros(1));archives[arm]={'archive_path':str(p),'archive_sha256':_sha(p)}
 pre={'runtime_code_sha256':code,'native_authority':native,'native_receipt_sha256':'c'*64}
 side=root/'authority_pre.json';side.write_text(json.dumps({'authority':pre}))
 receipt={'status':'PASS_COMPLETE_QUERYAGE_STREAM_EQUIVALENCE_ONLY','runtime_kind':'cached','authority_pre':pre,'authority_post':pre,'authority_pre_path':str(side),'authority_pre_sha256':_sha(side),'results':archives}
 p=root/'receipt.json';p.write_text(json.dumps(receipt));monkeypatch.setattr(c.complete,'runtime_code',lambda:code);monkeypatch.setattr(c.complete,'_native_authority',lambda *args: ({},native))
 bound=c._proof('cached',root,_sha(p))
 assert bound['native_authority']==native and set(bound['result_archives'])=={'FLAT','ROUTE'}

def test_audit_authority_uses_external_json_key_form():
 raw={'epoch_artifact_hashes':{'FLAT':{1:{'receipt_sha256':'x'}}},'selected_primary_ema_epoch':{'FLAT':2}}
 assert c._canonical(raw)==json.loads(json.dumps(raw,sort_keys=True))
 assert '1' in c._canonical(raw)['epoch_artifact_hashes']['FLAT']
