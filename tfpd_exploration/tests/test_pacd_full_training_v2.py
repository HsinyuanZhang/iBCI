"""No-data contract checks for the profile-composed PACD full V2 successor."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paired_anchored_calibration_dropout_full_v2 import plan,smoke
from src.paired_anchored_calibration_dropout_full_v1 import smoke as shared

def test_v2_profile_is_distinct_and_reuses_single_v1_executor():
 assert smoke.V2_EXECUTION_PROFILE.identity=="full-v2"
 assert smoke.V2_EXECUTION_PROFILE.plan.ARMS==plan.ARMS
 assert smoke.execute.__module__.endswith("full_v2.smoke")
def test_v2_public_cli_is_inert():
 result=subprocess.run([sys.executable,"-S",str(ROOT/"scripts/run_pacd_full_training_v2.py")],capture_output=True,text=True)
 assert result.returncode==0 and plan.CELL in result.stdout
 result=subprocess.run([sys.executable,"-S",str(ROOT/"scripts/run_pacd_full_training_v2.py"),"--execute"],capture_output=True,text=True)
 assert result.returncode==2

def test_v2_wrapper_shared_execute_synthetic_success_and_failure(tmp_path,monkeypatch):
 """V2 profile reaches the single shared lifecycle, never a copied loop."""
 receipt=shared.v1.load_stdlib_receipt_module(REPO:=ROOT.parent);gpu={"physical_index":"0"};pred={"v1_failure":"exact"};closure={"closure_sha256":"v2closure","files":{}}
 p=plan.RUNTIME_PLAN
 for key,value in {"ARMS":{"p0":{"short_m":30,"root":"v2out"}},"EPOCHS":4,"STEPS_PER_EPOCH":1,"TOTAL_STEPS":4,"FINAL_EPOCHS":(0,1,2,3),"SENTINELS":(0,),"BOUND_PATTERNS":(),"REVIEW_EVIDENCE_PATHS":(),"EXPECTED_INITIAL_STATE_SHA":"state","EXPECTED_BEHAVIOR_NORMALIZER_SHA":"b","EXPECTED_SIDE_NORMALIZER_SHA":"s","V2_THROUGHPUT_STEPS_PER_SECOND":{"p0":1.},"MECHANICAL_PROJECTED_HOURS":{"p0":1.}}.items(): monkeypatch.setattr(p,key,value,raising=False)
 monkeypatch.setattr(smoke,"validate_v1_failure",lambda root:pred);profile=shared.ExecutionProfile(identity="full-v2-test",plan=p,predecessor_validator=lambda root:pred)
 monkeypatch.setattr(shared.v1,"preflight_gpu0_idle",lambda:gpu);monkeypatch.setattr(shared.v1,"recheck_gpu0_after_attempt",lambda x:gpu);monkeypatch.setattr(shared.v1,"load_stdlib_receipt_module",lambda root:receipt);monkeypatch.setattr(shared.v1,"exact_source_closure",lambda root,receipt,patterns:closure);monkeypatch.setattr(shared.v1,"verify_expected_sealed_files",lambda root,receipt:{})
 class M(torch.nn.Module):
  def __init__(self): super().__init__();self.ps=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.)) for _ in range(29)])
  def forward(self,n,*,calib_trials,side_features): return n.mean(-1,keepdim=True).repeat(1,1,2),side_features.mean(-1)
 initial=tmp_path/"i.pt";torch.save({"state_dict":M().state_dict(),"state_sha256":"state"},initial);(tmp_path/"i.pt.sha256").write_text(f"{hashlib.sha256(initial.read_bytes()).hexdigest()}  i.pt\n")
 manifest=tmp_path/"manifest";manifest.write_text("m")
 class DM:
  train_dataset=type("D",(),{"sessions":["s"],"window_indices":[("s",0)]})();session_splits={"train":[str(i) for i in range(27)]};session_files={"val":[],"test":[]};_behavior_stats=(torch.tensor(0),torch.tensor(1));_side_feature_stats=(torch.tensor(0),torch.tensor(1))
 class A2:
  MANIFEST_PATH=manifest;EXPECTED_MANIFEST_SHA256=hashlib.sha256(b"m").hexdigest()
  normalizer_value_sha256=staticmethod(lambda *x:"b" if x[0] is DM._behavior_stats[0] else "s")
 class AR:
  PAD_VALUE=-1.;WINDOW_SIZE=50;build_datamodule=staticmethod(lambda args:(DM(),A2()))
  def seal_file(path):
   h=hashlib.sha256(path.read_bytes()).hexdigest();os.chmod(path,0o444);Path(str(path)+".sha256").write_text(f"{h}  {path.name}\n");os.chmod(Path(str(path)+".sha256"),0o444);return h
 class AC:
  ADAM_CONSTRUCTOR={"lr":1e-3,"betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"amsgrad":False};param_groups_by_branch=staticmethod(lambda m:(list(m.ps),list(m.ps)));state_sha256=staticmethod(lambda m:"state");optimizer_sha256=staticmethod(lambda o:"opt");t4_authority_fingerprint=staticmethod(lambda x:"t4");w_side_block=staticmethod(lambda m:torch.zeros(1))
 class PR: build_population_robustness_model=staticmethod(lambda **kw:M())
 class MS: build_swa_final_four=staticmethod(lambda paths,out:(torch.save(torch.load(paths[-1],weights_only=False),out) or {"ok":True}))
 class Cuda:
  synchronize=staticmethod(lambda d:None);memory_allocated=staticmethod(lambda d:0);max_memory_allocated=staticmethod(lambda d:0);max_memory_reserved=staticmethod(lambda d:0)
 class F:
  identity="v2-test"
  def after_attempt(self,*x):
   T=type("T",(),{"cuda":Cuda(),"backends":torch.backends,"load":staticmethod(torch.load),"save":staticmethod(torch.save),"optim":torch.optim,"count_nonzero":staticmethod(torch.count_nonzero),"set_num_threads":staticmethod(lambda x:None)})
   return type("R",(),{"torch":T(),"pl":type("P",(),{"seed_everything":staticmethod(lambda *x,**k:None)})(),"device":"cpu","stack":{"receipt":receipt,"arm_runner":AR,"pop_robust":PR,"arm_common":AC,"matched_scorer":MS}})()
  def sampler(self,*x,**k): return type("S",(),{"batched_indices":[[0]],"__len__":lambda s:1})()
  def loader(self,*x): return []
 f=F();monkeypatch.setattr(shared.v1,"_assert_cuda_binding_after_attempt",lambda t:"cpu");monkeypatch.setattr(shared,"run_epoch",lambda **kw:(kw["probe_sink"]((torch.randn(2,50,5),torch.randn(2,30,100,5),torch.randn(2,5,4))) if kw.get("probe_sink") else None) or {"epoch":kw["epoch"],"optimizer_steps":1});monkeypatch.setattr(shared,"_swa_runtime_proof",lambda **kw:{"state_before_sha256":"state","state_after_sha256":"state"})
 artifact=shared._ArtifactRoot(tmp_path,"p0",p);cap=shared._RootCapability(shared._CAPABILITY_SECRET,artifact=artifact,predecessor=pred,closure=closure,device=gpu,factory_identity="v2-test:full-v2-test");args=type("A",(),{"seed":42,"train_batch_size":32,"num_workers":0,"initial_state":initial})()
 assert smoke.execute(root=tmp_path,arm="p0",args=args,capability=cap,runtime_factory=f,profile=profile)==0
 out=tmp_path/"v2out";terminal=json.loads((out/"terminal.json").read_text());assert terminal["schema"].startswith("pacd_matched_full_training_v2") and len(terminal["epochs"])==4 and terminal["predecessor"]==pred
 with pytest.raises(RuntimeError,match="already consumed"): smoke.execute(root=tmp_path,arm="p0",args=args,capability=cap,runtime_factory=f,profile=profile)
 # A separate fresh root exercises the actual shared failure publisher through
 # the V2 wrapper; no Torch/source/checkpoint factory method is reached.
 failroot=tmp_path/"failure";failroot.mkdir();artifact=shared._ArtifactRoot(failroot,"p0",p);badcap=shared._RootCapability(shared._CAPABILITY_SECRET,artifact=artifact,predecessor=pred,closure=closure,device=gpu,factory_identity="v2-test:full-v2-test")
 monkeypatch.setattr(shared.v1,"recheck_gpu0_after_attempt",lambda x: (_ for _ in ()).throw(RuntimeError("V2 boundary")))
 with pytest.raises(RuntimeError,match="V2 boundary"): smoke.execute(root=failroot,arm="p0",args=args,capability=badcap,runtime_factory=f,profile=profile)
 failed=failroot/"v2out";assert (failed/"attempt.json").is_file() and (failed/"failure.json").is_file() and not (failed/"terminal.json").exists()
