"""No-data/no-CUDA contract tests for PACD matched full-training primitives."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent;sys.path.insert(0,str(ROOT))
from src.paired_anchored_calibration_dropout_full_v1 import plan,predecessor,runner,smoke

def _leaf(p,b):
 p.write_bytes(b);os.chmod(p,0o444);s=p.with_name(p.name+".sha256");s.write_text(f"{hashlib.sha256(b).hexdigest()}  {p.name}\n");os.chmod(s,0o444)
def _step(name):
 e={"dropout_pair_equal":True,"rng_short_transition_equal":True,"rng_pair_transition_equal":True,"parameter_finiteness":{"materialized":29,"skipped_uninitialized_lazy":2}}
 if name=="p0": e.update({"prediction_pair_equal":True,"identity_pair_equal":True})
 return e
def _v2(tmp,monkeypatch):
 d=tmp/"v2";d.mkdir();a={};ab=(json.dumps(a)+"\n").encode();ash=hashlib.sha256(ab).hexdigest();v1={"relative":"tfpd_exploration/results/paired_anchored_calibration_dropout_v1/smoke_seed42","attempt_sha256":"e1d6cd813b2bc49d89121b3341271a0bf94176abe3811d2f3b6f6ffa4a28b986","failure_sha256":"82ca6850cdd50f0b5ea5073eb89809a0c6ea426ac3f8f7d9e03fc44d1bd7e273","topology":["attempt.json","attempt.json.sha256","failure.json","failure.json.sha256"]};facts={"within_dev_sessions_opened":False,"external_sub_m_opened":False,"formal_or_organizer_held_data_opened":False,"scorer_called":False,"source_roster_n":27,"val_paths_resolved":[],"test_paths_resolved":[],"single_source_datamodule_materialization":True};t={"attempt_sha256":ash,"status":"PACD_SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE","source_only":True,"target_access":False,"predecessor":v1,"no_target_facts":facts,"source_closure":{"launch":{"closure_sha256":"c"},"final":{"closure_sha256":"c"}},"arms":[{"arm":{"name":n},"steps":[_step(n) for _ in range(8)]} for n in ("p0","p1","p2")]};tb=(json.dumps(t)+"\n").encode();tsh=hashlib.sha256(tb).hexdigest();_leaf(d/"attempt.json",ab);_leaf(d/"terminal.json",tb);monkeypatch.setattr(plan,"V2_RELATIVE","v2");monkeypatch.setattr(plan,"V2_ATTEMPT_SHA",ash);monkeypatch.setattr(plan,"V2_TERMINAL_SHA",tsh);monkeypatch.setattr(plan,"V2_CLOSURE_SHA","c");return d
def test_accepted_v2_held_graph_and_drift(tmp_path,monkeypatch):
 d=_v2(tmp_path,monkeypatch);assert predecessor.validate_v2_terminal(tmp_path,"v2")["topology"]==list(predecessor.LEAVES)
 (d/"extra").write_text("x")
 with pytest.raises(predecessor.PredecessorError): predecessor.validate_v2_terminal(tmp_path,"v2")
def test_v2_semantic_and_sidecar_adversaries_fail_closed(tmp_path,monkeypatch):
 d=_v2(tmp_path,monkeypatch)
 # Rebind the synthetic terminal literal so this tests semantics, not merely
 # the immutable-byte literal.
 body=json.loads((d/"terminal.json").read_text());body["arms"][0]["steps"][3]["prediction_pair_equal"]=False
 raw=(json.dumps(body)+"\n").encode();os.chmod(d/"terminal.json",0o644);os.chmod(d/"terminal.json.sha256",0o644);(d/"terminal.json").write_bytes(raw);_leaf(d/"terminal.json",raw);monkeypatch.setattr(plan,"V2_TERMINAL_SHA",hashlib.sha256(raw).hexdigest())
 with pytest.raises(predecessor.PredecessorError): predecessor.validate_v2_terminal(tmp_path,"v2")
def test_v2_symlink_and_mode_adversaries_fail_closed(tmp_path,monkeypatch):
 d=_v2(tmp_path,monkeypatch);os.chmod(d/"terminal.json",0o644)
 with pytest.raises(predecessor.PredecessorError): predecessor.validate_v2_terminal(tmp_path,"v2")
def test_full_dry_cli_is_inert():
 assert smoke.dry_payload()["no_torch_import"] is True and smoke.dry_payload()["epochs"]==48
class _Tensor:
 def to(self,d):return self
class _Cuda:
 def __init__(self):self.calls=[]
 def synchronize(self,d):self.calls.append(d)
class _Torch: 
 def __init__(self):self.cuda=_Cuda()
class _Opt: param_groups=[{}]
class _ArmCommon:
 @staticmethod
 def lr_at_step(step,epochs,spe):return step/100
def test_epoch_delegates_to_reviewed_paired_operator_and_chains(monkeypatch):
 monkeypatch.setattr(plan,"STEPS_PER_EPOCH",4);monkeypatch.setattr(plan,"SENTINELS",(0,1,2,3));calls=[]
 def fake(**kw):
  calls.append(kw["short_m"]);return {"loss_anchor":1.,"loss_short":2.,"loss_combined":1.5,"anchor_encoder_grad_norm":1.,"anchor_decoder_grad_norm":2.,"short_encoder_grad_norm":1.2,"short_decoder_grad_norm":2.2,"encoder_branch_gradient_cosine":.5,"decoder_branch_gradient_cosine":.5,"combined_encoder_grad_norm":3.,"combined_decoder_grad_norm":4.,"dropout":{"p":.2},"rng_before_sha256":"a","rng_after_pair_sha256":"b","calibration_anchor_sha256":"c","calibration_short_sha256":"d","calibration_full_sha256":"e","calibration_full_after_sha256":"e","rng_short_transition_equal":True,"rng_pair_transition_equal":True,"prediction_pair_equal":True,"identity_pair_equal":True,"valid_bins":3,"optimizer_steps":1,"parameter_finiteness":{"materialized":29,"skipped_uninitialized_lazy":2}}
 monkeypatch.setattr(runner,"paired_train_step",fake)
 batch=(_Tensor(),_Tensor(),_Tensor(),["s"],_Tensor());out=runner.run_epoch(model=object(),optimizer=_Opt(),loader=[batch]*4,arm={"name":"p0","short_m":30},device="cpu",torch=_Torch(),arm_common=_ArmCommon(),encoder_parameters=[],decoder_parameters=[],epoch=0,pad_value=-1.,steps_per_epoch=4)
 assert calls==[30]*4 and out["optimizer_steps"]==4 and out["p0_prediction_mismatches"]==0 and len(out["sentinels"])==4
def test_epoch_rejects_count_and_evidence_drift(monkeypatch):
 monkeypatch.setattr(plan,"STEPS_PER_EPOCH",2);monkeypatch.setattr(plan,"SENTINELS",(0,1))
 def bad(**kw): return {"optimizer_steps":1}
 monkeypatch.setattr(runner,"paired_train_step",bad);batch=(_Tensor(),_Tensor(),_Tensor(),["s"],_Tensor())
 with pytest.raises(runner.FullInvariantError): runner.run_epoch(model=object(),optimizer=_Opt(),loader=[batch]*2,arm={"name":"p1","short_m":4},device="cpu",torch=_Torch(),arm_common=_ArmCommon(),encoder_parameters=[],decoder_parameters=[],epoch=0,pad_value=-1.,steps_per_epoch=2)
def test_full_cli_requires_opaque_authorization_subprocess():
 import subprocess
 r=subprocess.run([sys.executable,"-S",str(ROOT/"scripts/run_pacd_full_training_v1.py"),"--execute"],capture_output=True,text=True)
 assert r.returncode==2
def test_public_cli_execute_is_fail_closed_even_with_old_flag():
 import subprocess
 r=subprocess.run([sys.executable,"-S",str(ROOT/"scripts/run_pacd_full_training_v1.py"),"--execute","--opaque-root-authorization"],capture_output=True,text=True)
 assert r.returncode==2
def test_internal_capability_and_fixed_science_reject_before_predecessor(tmp_path):
 cap=object()
 for field,value in (("seed",7),("train_batch_size",16),("num_workers",1)):
  a=type("Args",(),{"seed":42,"train_batch_size":32,"num_workers":0})();setattr(a,field,value)
  with pytest.raises(RuntimeError): smoke.execute(root=tmp_path,arm="p0",args=a,capability=cap)
def test_temp_success_receipt_lifecycle_and_exclusive_failure(tmp_path):
 receipt=smoke.v1.load_stdlib_receipt_module(REPO)
 out=tmp_path/"p1_m4_seed42";out.mkdir()
 attempt=smoke._write_and_describe(receipt,out/"attempt.json",{"status":"attempt"})
 launch=smoke._write_and_describe(receipt,out/"launch.json",{"attempt":attempt})
 authority=smoke._write_and_describe(receipt,out/"source_authority.json",{"attempt":attempt,"target_access":False})
 epochs=[smoke._write_and_describe(receipt,out/f"epoch{i:03d}.json",{"epoch":i,"steps":2}) for i in range(48)]
 checkpoints=[]
 for epoch in (44,45,46,47):
  p=out/f"epoch{epoch:03d}.pt";p.write_bytes(f"checkpoint-{epoch}".encode());os.chmod(p,0o444)
  side=Path(str(p)+".sha256");side.write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n");os.chmod(side,0o444);checkpoints.append(smoke._sealed_descriptor(receipt,p,epoch=epoch))
 swa=out/"swa_final4.pt";swa.write_bytes(b"swa");os.chmod(swa,0o444);(out/"swa_final4.pt.sha256").write_text(f"{hashlib.sha256(b'swa').hexdigest()}  swa_final4.pt\n");os.chmod(out/"swa_final4.pt.sha256",0o444);swa_d=smoke._sealed_descriptor(receipt,swa)
 manifest=smoke._write_and_describe(receipt,out/"manifest.json",{"checkpoints":checkpoints,"swa":swa_d})
 terminal=smoke._write_and_describe(receipt,out/"terminal.json",{"attempt":attempt,"launch":launch,"source_authority":authority,"epochs":epochs,"checkpoints":checkpoints,"swa":swa_d,"manifest":manifest})
 expected={"attempt.json","attempt.json.sha256","launch.json","launch.json.sha256","source_authority.json","source_authority.json.sha256","manifest.json","manifest.json.sha256","terminal.json","terminal.json.sha256","swa_final4.pt","swa_final4.pt.sha256"}
 expected|={f"epoch{i:03d}.json" for i in range(48)}|{f"epoch{i:03d}.json.sha256" for i in range(48)}
 expected|={f"epoch{i:03d}.pt" for i in (44,45,46,47)}|{f"epoch{i:03d}.pt.sha256" for i in (44,45,46,47)}
 assert len(epochs)==48 and [x["epoch"] for x in checkpoints]==[44,45,46,47] and terminal["sha256"] and {p.name for p in out.iterdir()}==expected
 for path in out.iterdir(): assert os.stat(path).st_mode & 0o777 == 0o444
 with pytest.raises(RuntimeError): smoke.v1.publish_failure(out,receipt,{"status":"bad"})
def test_sealed_descriptor_rejects_mode_and_symlink(tmp_path):
 receipt=smoke.v1.load_stdlib_receipt_module(REPO);p=tmp_path/"artifact.bin";p.write_bytes(b"x");os.chmod(p,0o444);s=Path(str(p)+".sha256");s.write_text(f"{hashlib.sha256(b'x').hexdigest()}  artifact.bin\n");os.chmod(s,0o444)
 assert smoke._sealed_descriptor(receipt,p)["sha256"]
 os.chmod(s,0o644)
 with pytest.raises(RuntimeError): smoke._sealed_descriptor(receipt,p)
def test_sampler_order_digest_rejects_reorder_and_window_mutation():
 ds=type("DS",(),{"window_indices":[("s",0),("s",1)]})();sampler=type("Sampler",(),{"batched_indices":[[0,1]]})()
 stable=smoke._sampler_order_evidence(ds,sampler);sampler.batched_indices=[[1,0]]
 assert smoke._sampler_order_evidence(ds,sampler)!=stable
 sampler.batched_indices=[[0,1]];ds.window_indices=[("s",9),("s",1)]
 assert smoke._sampler_order_evidence(ds,sampler)!=stable
def test_actual_shaped_cpu_swa_runtime_proof_exercises_production_helper():
 class CellLike(torch.nn.Module):
  def __init__(self):
   super().__init__();self.live=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0)) for _ in range(29)]);self.register_parameter("lazy1",torch.nn.parameter.UninitializedParameter());self.register_parameter("lazy2",torch.nn.parameter.UninitializedParameter())
  def forward(self,neural,*,calib_trials,side_features):
   value=neural.mean(dim=-1,keepdim=True)*self.live[0];return value.repeat(1,1,2),side_features.mean(dim=-1)
 class ArmCommon:
  @staticmethod
  def state_sha256(model): return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.live)).hexdigest()
 model=CellLike().eval();probe=(torch.randn(2,50,5),torch.randn(2,30,100,5),torch.randn(2,5,4))
 proof=smoke._swa_runtime_proof(torch=torch,model=model,probe=probe,device="cpu",arm_common=ArmCommon())
 assert proof["repeated_forward_bitwise_equal"] and proof["output_finite"] and proof["dynamic_dropout_calls"]==0 and proof["parameter_finiteness"]=={"materialized":29,"skipped_uninitialized_lazy":2}
def test_real_execute_publishes_attempt_then_exclusive_honest_failure(tmp_path,monkeypatch):
 """Exercise production execute ordering without importing Torch or data."""
 receipt=smoke.v1.load_stdlib_receipt_module(REPO);gpu={"physical_index":"0"};pred={"lineage":"ok"};closure={"closure_sha256":"c","files":{}}
 monkeypatch.setattr(plan,"ARMS",{"p0":{"short_m":30,"root":"out"}});monkeypatch.setattr(plan,"BOUND_PATTERNS",());monkeypatch.setattr(plan,"REVIEW_EVIDENCE_PATHS",())
 monkeypatch.setattr(smoke,"validate_v2_terminal",lambda root:pred);monkeypatch.setattr(smoke.v1,"preflight_gpu0_idle",lambda:gpu);monkeypatch.setattr(smoke.v1,"load_stdlib_receipt_module",lambda root:receipt);monkeypatch.setattr(smoke.v1,"exact_source_closure",lambda root,receipt,patterns:closure)
 monkeypatch.setattr(smoke.v1,"recheck_gpu0_after_attempt",lambda value: (_ for _ in ()).throw(RuntimeError("runtime boundary")))
 artifact=smoke._ArtifactRoot(tmp_path,"p0");cap=smoke._RootCapability(smoke._CAPABILITY_SECRET,artifact=artifact,predecessor=pred,closure=closure,device=gpu,factory_identity="test:full-v1")
 args=type("Args",(),{"seed":42,"train_batch_size":32,"num_workers":0})()
 class Factory: identity="test"
 with pytest.raises(RuntimeError,match="runtime boundary"): smoke.execute(root=tmp_path,arm="p0",args=args,capability=cap,runtime_factory=Factory())
 out=tmp_path/"out";assert (out/"attempt.json").is_file() and (out/"failure.json").is_file() and not (out/"terminal.json").exists()
 failure=json.loads((out/"failure.json").read_text());assert failure["progress"]["torch_import_attempted"] is False and failure["progress"]["source_resolve_attempted"] is False
 with pytest.raises(RuntimeError,match="already consumed"): smoke.execute(root=tmp_path,arm="p0",args=args,capability=cap,runtime_factory=Factory())
def test_real_execute_synthetic_success_runs_full_receipt_lifecycle(tmp_path,monkeypatch):
 """A typed test backend drives production execute through terminal receipt."""
 receipt=smoke.v1.load_stdlib_receipt_module(REPO);gpu={"physical_index":"0"};pred={"lineage":"ok"};closure={"closure_sha256":"c","files":{}}
 monkeypatch.setattr(plan,"ARMS",{"p0":{"short_m":30,"root":"out"}});monkeypatch.setattr(plan,"BOUND_PATTERNS",());monkeypatch.setattr(plan,"REVIEW_EVIDENCE_PATHS",());monkeypatch.setattr(plan,"EPOCHS",4);monkeypatch.setattr(plan,"STEPS_PER_EPOCH",1);monkeypatch.setattr(plan,"TOTAL_STEPS",4);monkeypatch.setattr(plan,"FINAL_EPOCHS",(0,1,2,3));monkeypatch.setattr(plan,"SENTINELS",(0,));monkeypatch.setattr(plan,"EXPECTED_INITIAL_STATE_SHA","synthetic-state");monkeypatch.setattr(plan,"EXPECTED_BEHAVIOR_NORMALIZER_SHA","behavior");monkeypatch.setattr(plan,"EXPECTED_SIDE_NORMALIZER_SHA","side");monkeypatch.setattr(plan,"V2_THROUGHPUT_STEPS_PER_SECOND",{"p0":1.});monkeypatch.setattr(plan,"MECHANICAL_PROJECTED_HOURS",{"p0":1.})
 monkeypatch.setattr(smoke,"validate_v2_terminal",lambda root:pred);monkeypatch.setattr(smoke.v1,"preflight_gpu0_idle",lambda:gpu);monkeypatch.setattr(smoke.v1,"recheck_gpu0_after_attempt",lambda value:gpu);monkeypatch.setattr(smoke.v1,"load_stdlib_receipt_module",lambda root:receipt);monkeypatch.setattr(smoke.v1,"exact_source_closure",lambda root,receipt,patterns:closure);monkeypatch.setattr(smoke.v1,"verify_expected_sealed_files",lambda root,receipt:{})
 class CellLike(torch.nn.Module):
  def __init__(self):
   super().__init__();self.live=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0)) for _ in range(29)])
  def forward(self,neural,*,calib_trials,side_features): return neural.mean(-1,keepdim=True).repeat(1,1,2),side_features.mean(-1)
 def digest(model): return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.live)).hexdigest()
 initial=tmp_path/"initial.pt";seed_model=CellLike();torch.save({"state_dict":seed_model.state_dict(),"state_sha256":"synthetic-state"},initial);(tmp_path/"initial.pt.sha256").write_text(f"{hashlib.sha256(initial.read_bytes()).hexdigest()}  initial.pt\n")
 class DM:
  train_dataset=type("DS",(),{"sessions":["s"],"window_indices":[("s",0)]})()
  session_splits={"train":[f"s{i}" for i in range(27)]};session_files={"val":[],"test":[]};_behavior_stats=(torch.tensor(0.),torch.tensor(1.));_side_feature_stats=(torch.tensor(0.),torch.tensor(1.))
 manifest=tmp_path/"manifest";manifest.write_text("m")
 class A2:
  MANIFEST_PATH=manifest;EXPECTED_MANIFEST_SHA256=hashlib.sha256(b"m").hexdigest()
  @staticmethod
  def normalizer_value_sha256(*x): return "behavior" if len(x)==2 and x[0] is DM._behavior_stats[0] else "side"
 class AR: PAD_VALUE=-1.;WINDOW_SIZE=50
 class PR:
  @staticmethod
  def build_population_robustness_model(**kw): return CellLike()
 class AC:
  ADAM_CONSTRUCTOR={"lr":1e-3,"betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"amsgrad":False}
  @staticmethod
  def param_groups_by_branch(m): return list(m.live),list(m.live)
  @staticmethod
  def state_sha256(m): return "synthetic-state"
  @staticmethod
  def optimizer_sha256(o): return "optimizer"
  @staticmethod
  def t4_authority_fingerprint(x): return "t4"
  @staticmethod
  def w_side_block(m): return torch.zeros(1)
 class MS:
  @staticmethod
  def build_swa_final_four(paths,out): torch.save(torch.load(paths[-1],weights_only=False),out);return {"inherited":True}
 stack={"receipt":receipt,"arm_runner":AR(),"pop_robust":PR(),"arm_common":AC(),"matched_scorer":MS()}
 class Cuda:
  @staticmethod
  def reset_peak_memory_stats(d): pass
  @staticmethod
  def memory_allocated(d): return 0
  @staticmethod
  def max_memory_allocated(d): return 0
  @staticmethod
  def max_memory_reserved(d): return 0
  @staticmethod
  def synchronize(d): pass
 class Factory:
  identity="synthetic-execute";production=False
  def after_attempt(self,root,g): return type("R",(),{"torch":type("T",(),{"set_num_threads":staticmethod(lambda n:None),"cuda":Cuda(),"backends":torch.backends,"load":staticmethod(torch.load),"save":staticmethod(torch.save),"optim":torch.optim,"no_grad":staticmethod(torch.no_grad),"equal":staticmethod(torch.equal),"isfinite":staticmethod(torch.isfinite),"count_nonzero":staticmethod(torch.count_nonzero)})(),"pl":type("P",(),{"seed_everything":staticmethod(lambda *a,**k:None)})(),"device":"cpu","stack":stack})()
  def sampler(self,ds,**kw): return type("Sampler",(),{"batched_indices":[[0]],"__len__":lambda self:1})()
  def loader(self,ds,sampler): return []
 factory=Factory();monkeypatch.setattr(smoke.v1,"_assert_cuda_binding_after_attempt",lambda t:"cpu");monkeypatch.setattr(smoke,"run_epoch",lambda **kw: (kw["probe_sink"]((torch.randn(2,50,5),torch.randn(2,30,100,5),torch.randn(2,5,4))) if kw.get("probe_sink") else None) or {"epoch":kw["epoch"],"optimizer_steps":1,"wall_seconds":0.1})
 AR.build_datamodule=staticmethod(lambda args:(DM(),A2()));monkeypatch.setattr(smoke,"_swa_runtime_proof",lambda **kw:{"synthetic":True,"output_finite":True,"state_before_sha256":"synthetic-state","state_after_sha256":"synthetic-state"})
 def seal(path):
  digest_value=hashlib.sha256(path.read_bytes()).hexdigest();os.chmod(path,0o444);Path(str(path)+".sha256").write_text(f"{digest_value}  {path.name}\n");os.chmod(Path(str(path)+".sha256"),0o444);return digest_value
 AR.seal_file=staticmethod(seal);artifact=smoke._ArtifactRoot(tmp_path,"p0");cap=smoke._RootCapability(smoke._CAPABILITY_SECRET,artifact=artifact,predecessor=pred,closure=closure,device=gpu,factory_identity=factory.identity+":full-v1");args=type("Args",(),{"seed":42,"train_batch_size":32,"num_workers":0,"initial_state":initial})()
 assert smoke.execute(root=tmp_path,arm="p0",args=args,capability=cap,runtime_factory=factory)==0
 out=tmp_path/"out";term=json.loads((out/"terminal.json").read_text());assert len(term["epochs"])==4 and len(term["checkpoints"])==4 and term["launch"]["sha256"] and term["swa"]["proof"]["output_finite"]
