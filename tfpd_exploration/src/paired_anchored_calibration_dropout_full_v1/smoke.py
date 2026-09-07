"""Full PACD lifecycle, reusing V1 paired-step science and sealed helpers."""
from __future__ import annotations
import hashlib,json,os,stat,time,traceback
from pathlib import Path
from typing import Any,Callable
from dataclasses import dataclass
from . import plan
from .predecessor import validate_v2_terminal
from .runner import run_epoch
from src.paired_anchored_calibration_dropout_v1 import smoke as v1
_CAPABILITY_SECRET=object()
@dataclass(frozen=True)
class ExecutionProfile:
 identity:str
 plan:Any
 predecessor_validator:Callable[[Path],dict]
V1_EXECUTION_PROFILE=ExecutionProfile(identity="full-v1",plan=plan,predecessor_validator=lambda root: validate_v2_terminal(root))
class _ProductionRuntimeFactory:
 """Only production backend: sealed loaders/model and reviewed V1 operator."""
 identity="production-sealed-v1-runtime"; production=True
 def after_attempt(self,root,gpu):
  import torch,lightning.pytorch as pl
  torch.set_num_threads(1);device=v1._assert_cuda_binding_after_attempt(torch);torch.cuda.reset_peak_memory_stats(device)
  return type("Runtime",(),{"torch":torch,"pl":pl,"device":device,"stack":v1._load_execution_stack(root)})()
 def sampler(self,dataset,*,batch_size,seed):
  from mc_maze.multisession_datamodule import SessionBatchSampler
  return SessionBatchSampler(dataset,batch_size=batch_size,shuffle=True,seed=seed)
 def loader(self,dataset,sampler):
  from torch.utils.data import DataLoader
  return DataLoader(dataset,batch_sampler=sampler,num_workers=0,pin_memory=True)
PRODUCTION_RUNTIME_FACTORY=_ProductionRuntimeFactory()
class _ArtifactRoot:
 def __init__(self,root:Path,arm:str,p=plan):
  self.root=root.absolute();self.arm=arm;self.plan=p;self.relative=p.ARMS[arm]["root"];self.out=self.root/self.relative
  self.parents=[]
  current=self.root
  for part in Path(self.relative).parts:
   current=current/part
   if current.exists():
    st=os.lstat(current);self.parents.append((str(current),st.st_dev,st.st_ino))
 def validate(self,*,fresh:bool)->Path:
  _req(self.arm in self.plan.ARMS,"artifact arm drift")
  expected=self.root/self.plan.ARMS[self.arm]["root"]
  _req(expected.absolute()==self.out.absolute(),"artifact root identity drift")
  for path,dev,ino in self.parents:
   st=os.lstat(path);_req((st.st_dev,st.st_ino)==(dev,ino),"artifact parent replacement")
  if fresh: v1.require_canonical_fresh_out_dir(self.root,self.out,self.relative)
  else: _req(self.out.is_dir() and not self.out.is_symlink(),"published artifact root drift")
  return self.out
class _RootCapability:
 def __init__(self,secret,*,artifact,predecessor,closure,device,factory_identity):
  if secret is not _CAPABILITY_SECRET: raise RuntimeError("root capability is nonconstructible")
  self.artifact=artifact; self.predecessor=predecessor; self.closure=closure; self.device=device;self.factory_identity=factory_identity;self.consumed=False
def _issue_root_capability_after_preflight(*,root:Path,arm:str,runtime_factory=PRODUCTION_RUNTIME_FACTORY,profile=V1_EXECUTION_PROFILE):
 """Root-only admission: bind live lineage, closure, and GPU profile once.

 The public CLI cannot call this seam.  Execute repeats every mutable-risk
 check and exact-compares these values, so a caller cannot substitute a root,
 closure, predecessor, or selected GPU after admission.
 """
 p=profile.plan;_req(arm in p.ARMS,"unknown arm");artifact=_ArtifactRoot(root,arm,p);artifact.validate(fresh=True);predecessor=profile.predecessor_validator(root)
 gpu=v1.preflight_gpu0_idle(); receipt=v1.load_stdlib_receipt_module(root)
 closure=v1.exact_source_closure(root,receipt,p.BOUND_PATTERNS)
 return _RootCapability(_CAPABILITY_SECRET,artifact=artifact,predecessor=predecessor,closure=closure,device=gpu,factory_identity=runtime_factory.identity+":"+profile.identity)

def dry_payload(): return {"cell":plan.CELL,"schema":plan.SCHEMA,"arms":plan.ARMS,"epochs":plan.EPOCHS,"steps_per_epoch":plan.STEPS_PER_EPOCH,"target_access":False,"no_torch_import":True,"full_launch_authorized":False}
def _req(x,m):
 if not x: raise RuntimeError(m)
def _root(root,arm,p=plan):
 out=root/p.ARMS[arm]["root"]; v1.require_canonical_fresh_out_dir(root,out,p.ARMS[arm]["root"]);return out
def _sha(receipt,path:Path)->str:
 return receipt.sha256_file(path)
def _canonical_json_sha(value:Any)->str:
 return hashlib.sha256(json.dumps(value,sort_keys=False,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def _sampler_order_evidence(dataset,sampler)->dict[str,str]:
 """Cheap source-order proof: indices only, never neural/behavior tensors."""
 windows=[[str(session),int(start)] for session,start in dataset.window_indices]
 batches=[[int(index) for index in batch] for batch in sampler.batched_indices]
 return {"window_indices_sha256":_canonical_json_sha(windows),"batched_indices_sha256":_canonical_json_sha(batches)}
def _sealed_descriptor(receipt,path:Path,**extra:Any)->dict[str,Any]:
 sidecar=Path(str(path)+".sha256")
 _req(path.is_file() and sidecar.is_file(),f"missing sealed artifact {path.name}")
 for item in (path,sidecar):
  mode=os.lstat(item).st_mode
  _req(stat.S_ISREG(mode) and stat.S_IMODE(mode)==0o444,f"immutable mode/symlink drift {item.name}")
 digest=_sha(receipt,path); _req(sidecar.read_text()==f"{digest}  {path.name}\n",f"sidecar drift {path.name}")
 return {"name":path.name,"sha256":digest,"sidecar":sidecar.name,**extra}
def _write_and_describe(receipt,path:Path,payload:dict[str,Any],**extra:Any)->dict[str,Any]:
 receipt.write_receipt_transactionally(path,payload); return _sealed_descriptor(receipt,path,**extra)
def _validate_capability(capability:Any,*,root:Path,arm:str,predecessor:dict[str,Any],closure:dict[str,Any],gpu:dict[str,Any],fresh:bool,runtime_factory,profile=V1_EXECUTION_PROFILE)->None:
 _req(isinstance(capability,_RootCapability),"root capability required")
 _req(capability.artifact.root==root.absolute() and capability.artifact.arm==arm,"capability arm/root drift")
 capability.artifact.validate(fresh=fresh)
 _req(capability.predecessor==predecessor,"capability predecessor drift")
 _req(capability.closure==closure,"capability closure drift")
 _req(capability.device==gpu,"capability device profile drift")
 _req(capability.factory_identity==runtime_factory.identity+":"+profile.identity,"capability runtime-factory drift")
def _output_equal(torch,left,right)->bool:
 if isinstance(left,tuple): return len(left)==len(right) and all(_output_equal(torch,a,b) for a,b in zip(left,right))
 return bool(torch.equal(left,right))
def _output_finite(torch,value)->bool:
 if isinstance(value,tuple): return all(_output_finite(torch,item) for item in value)
 return bool(torch.isfinite(value).all().item())
def _swa_runtime_proof(*,torch,model,probe,device,arm_common)->dict[str,Any]:
 """Prove strict-loaded SWA is finite, stateless, deterministic and eval-only."""
 _req(not model.training,"SWA must be eval/no-dropout")
 before=arm_common.state_sha256(model)
 neural,calibration,side=probe
 from src.paired_anchored_calibration_dropout_v1.core import assert_finite_materialized_parameters,record_unit_dropout_masks
 with record_unit_dropout_masks() as dropout_calls,torch.no_grad():
  first=model(neural.to(device),calib_trials=calibration.to(device),side_features=side.to(device))
  second=model(neural.to(device),calib_trials=calibration.to(device),side_features=side.to(device))
 _req(_output_equal(torch,first,second),"SWA repeated forward drift")
 _req(_output_finite(torch,first),"SWA output non-finite")
 _req(dropout_calls==[],"SWA eval dynamically dropped units")
 after=arm_common.state_sha256(model)
 _req(before==after,"SWA forward mutated state")
 finite=assert_finite_materialized_parameters(model)
 _req(finite=={"materialized":29,"skipped_uninitialized_lazy":2},"SWA parameter topology")
 return {"strict_load":True,"eval":True,"no_grad":True,"repeated_forward_bitwise_equal":True,"output_finite":True,"dynamic_dropout_calls":0,"state_before_sha256":before,"state_after_sha256":after,"parameter_finiteness":finite}
def execute(*,root:Path,arm:str,args:Any,capability:Any=None,runtime_factory=PRODUCTION_RUNTIME_FACTORY,profile=V1_EXECUTION_PROFILE)->int:
 plan=profile.plan
 _req(getattr(runtime_factory,"identity",None),"untyped runtime factory");_req(arm in plan.ARMS,"unknown arm");_req(args.seed==42,"seed");_req(args.train_batch_size==32,"batch");_req(args.num_workers==0,"num_workers");_req(isinstance(capability,_RootCapability),"root capability required");_req(not capability.consumed,"root capability already consumed"); predecessor=profile.predecessor_validator(root);out=_root(root,arm,plan);gpu=v1.preflight_gpu0_idle();receipt=v1.load_stdlib_receipt_module(root);closure=v1.exact_source_closure(root,receipt,plan.BOUND_PATTERNS);_validate_capability(capability,root=root,arm=arm,predecessor=predecessor,closure=closure,gpu=gpu,fresh=True,runtime_factory=runtime_factory,profile=profile);review={p:receipt.sha256_file(root/p) for p in plan.REVIEW_EVIDENCE_PATHS};capability.consumed=True;out.mkdir(parents=True)
 attempt={"schema":plan.SCHEMA+"_attempt","status":"ATTEMPT_PUBLISHED","cell":plan.CELL,"arm":arm,"arm_identity":plan.ARMS[arm],"budget":{"epochs":plan.EPOCHS,"steps_per_epoch":plan.STEPS_PER_EPOCH,"total_steps":plan.TOTAL_STEPS,"seed":plan.SEED,"batch_size":plan.BATCH_SIZE,"num_workers":0},"v2_throughput_precursor_steps_per_second":plan.V2_THROUGHPUT_STEPS_PER_SECOND[arm],"mechanical_projected_hours":plan.MECHANICAL_PROJECTED_HOURS[arm],"projection_is_not_stop_or_selection_rule":True,"predecessor":predecessor,"source_closure":closure,"review_evidence":review,"device":gpu,"target_access":False,"ordering":"before torch/source/checkpoint/cuda"};attempt_descriptor=_write_and_describe(receipt,out/"attempt.json",attempt);attempt_sha=attempt_descriptor["sha256"];started=time.monotonic(); stack=None; published={"attempt":attempt_descriptor}; progress={"torch_import_attempted":False,"torch_imported":False,"cuda_bind_attempted":False,"cuda_bound":False,"source_resolve_attempted":False,"source_opened":False,"checkpoint_deserialize_attempted":False,"checkpoint_deserialized":False,"epochs_published":0,"checkpoints_published":0,"swa_published":False}
 try:
  gpu2=v1.recheck_gpu0_after_attempt(gpu);_req(gpu2==capability.device,"post-attempt device capability drift");progress["torch_import_attempted"]=True;progress["cuda_bind_attempted"]=True;runtime=runtime_factory.after_attempt(root,gpu2);torch=runtime.torch;pl=runtime.pl;device=runtime.device;stack=runtime.stack;progress["torch_imported"]=True;progress["cuda_bound"]=True;sealed=v1.verify_expected_sealed_files(root,stack["receipt"]); ar=stack["arm_runner"];pr=stack["pop_robust"];ac=stack["arm_common"];ms=stack["matched_scorer"]
  progress["source_resolve_attempted"]=True;dm,a2=ar.build_datamodule(args);progress["source_opened"]=True;_req(len(dm.session_splits["train"])==27 and dm.session_files["val"]==[] and dm.session_files["test"]==[],"source roster")
  ds=dm.train_dataset;pl.seed_everything(args.seed,workers=True);initial=Path(args.initial_state);sidecar=Path(str(initial)+".sha256");_req(initial.is_file() and sidecar.is_file() and receipt.sha256_file(initial)==sidecar.read_text().split()[0],"initial")
  progress["checkpoint_deserialize_attempted"]=True;payload=torch.load(initial,map_location="cpu",weights_only=False);progress["checkpoint_deserialized"]=True;model=pr.build_population_robustness_model(seed=args.seed,cell="D");model.load_state_dict(payload["state_dict"],strict=True);loaded_state_sha=ac.state_sha256(model);_req(loaded_state_sha==payload["state_sha256"]==plan.EXPECTED_INITIAL_STATE_SHA,"strict initial state sha");w_side=ac.w_side_block(model);_req(int(torch.count_nonzero(w_side).item())==0 and not bool(w_side.signbit().any().item()),"strict initial W_side zero");model.to(device).train();enc,dec=ac.param_groups_by_branch(model);opt=torch.optim.Adam(model.parameters(),lr=ac.ADAM_CONSTRUCTOR["lr"],betas=tuple(ac.ADAM_CONSTRUCTOR["betas"]),eps=ac.ADAM_CONSTRUCTOR["eps"],weight_decay=ac.ADAM_CONSTRUCTOR["weight_decay"],amsgrad=ac.ADAM_CONSTRUCTOR["amsgrad"])
  sampler=runtime_factory.sampler(ds,batch_size=args.train_batch_size,seed=args.seed);_req(len(sampler)==plan.STEPS_PER_EPOCH,"steps/epoch");order_evidence=_sampler_order_evidence(ds,sampler);loader=runtime_factory.loader(ds,sampler);behavior_mean,behavior_std=dm._behavior_stats;side_mean,side_std=dm._side_feature_stats;behavior_semantic=a2.normalizer_value_sha256(behavior_mean,behavior_std);side_semantic=a2.normalizer_value_sha256(side_mean,side_std);_req(behavior_semantic==plan.EXPECTED_BEHAVIOR_NORMALIZER_SHA,"behavior normalizer authority");_req(side_semantic==plan.EXPECTED_SIDE_NORMALIZER_SHA,"side normalizer authority");manifest_sha=receipt.sha256_file(a2.MANIFEST_PATH);_req(manifest_sha==a2.EXPECTED_MANIFEST_SHA256,"manifest authority");source_authority=_write_and_describe(receipt,out/"source_authority.json",{"ordered_source_roster":list(dm.session_splits["train"]),"source_roster_n":27,"val":[],"test":[],"target_access":False,"attempt_sha256":attempt_sha,"manifest":{"path":str(a2.MANIFEST_PATH),"sha256":manifest_sha},"normalizers":{"behavior_semantic_sha256":behavior_semantic,"side_feature_semantic_sha256":side_semantic},"t4_authority_sha256":ac.t4_authority_fingerprint(ds.sessions),"window":{"window_size":ar.WINDOW_SIZE,"calibration_n_trials":30,"max_trial_length":100,"bin_size_ms":20},"sampler":{"class":"SessionBatchSampler","batch_size":32,"shuffle":True,"seed":42,"num_workers":0,"steps_per_epoch":plan.STEPS_PER_EPOCH,**order_evidence},"initial_state":{"body_sha256":receipt.sha256_file(initial),"state_sha256":payload["state_sha256"],"strict_loaded_state_sha256":loaded_state_sha,"w_side_exact_positive_zero":True}});published["source_authority"]=source_authority;launch=_write_and_describe(receipt,out/"launch.json",{"schema":plan.SCHEMA+"_launch","attempt":attempt_descriptor,"source_authority":source_authority,"device_pre":gpu,"device_post":gpu2,"target_access":False,"tf32":{"matmul_allow_tf32":bool(torch.backends.cuda.matmul.allow_tf32),"cudnn_allow_tf32":bool(torch.backends.cudnn.allow_tf32)},"logical_device":str(device),"runtime_factory":runtime_factory.identity});published["launch"]=launch
  epochs=[];checks=[];swa_probe=[]
  for epoch in range(plan.EPOCHS):
   before_order=_sampler_order_evidence(ds,sampler);_req(before_order==order_evidence,"sampler order drift before epoch");epoch_kwargs={"model":model,"optimizer":opt,"loader":loader,"arm":{"name":arm,**plan.ARMS[arm]},"device":device,"torch":torch,"arm_common":ac,"encoder_parameters":enc,"decoder_parameters":dec,"epoch":epoch,"pad_value":ar.PAD_VALUE,"probe_sink":(lambda value: swa_probe.append(value)) if epoch==0 else None};zero_encoder_policy=getattr(plan,"ZERO_ENCODER_POLICY","reject");epoch_kwargs.update({"zero_encoder_policy":zero_encoder_policy} if zero_encoder_policy!="reject" else {});e=run_epoch(**epoch_kwargs);after_order=_sampler_order_evidence(ds,sampler);_req(after_order==order_evidence,"sampler order drift after epoch");e.update({"model_state_sha256":ac.state_sha256(model),"optimizer_state_sha256":ac.optimizer_sha256(opt),"adam_finite":v1._finite_optimizer_state(torch,opt),"sampler_order":order_evidence,"cuda":{"current_allocated":int(torch.cuda.memory_allocated(device)),"peak_allocated":int(torch.cuda.max_memory_allocated(device)),"peak_reserved":int(torch.cuda.max_memory_reserved(device))}});_req(e["adam_finite"],"adam")
   if epoch in plan.FINAL_EPOCHS:
    cp=out/f"epoch{epoch:03d}.pt";torch.save({"arm":arm,"epoch":epoch,"state_dict":model.state_dict(),"model_state_sha256":e["model_state_sha256"],"optimizer_state_sha256":e["optimizer_state_sha256"],"predecessor":predecessor,"implementation":{"closure_sha256":closure["closure_sha256"],"cell":plan.CELL}},cp);ar.seal_file(cp);e["checkpoint"]=_sealed_descriptor(receipt,cp,epoch=epoch,model_state_sha256=e["model_state_sha256"],optimizer_state_sha256=e["optimizer_state_sha256"],predecessor=predecessor,implementation_closure_sha256=closure["closure_sha256"]);checks.append(cp);progress["checkpoints_published"]+=1
   epoch_descriptor=_write_and_describe(receipt,out/f"epoch{epoch:03d}.json",e);epochs.append(epoch_descriptor);published["epochs"]=epochs;published["checkpoints"]=[_sealed_descriptor(receipt,p,epoch=int(p.stem[-3:])) for p in checks];progress["epochs_published"]+=1
  _req(progress["epochs_published"]==plan.EPOCHS and progress["checkpoints_published"]==4,"artifact count")
  _req(len(swa_probe)==1,"SWA probe topology")
  swa=out/"swa_final4.pt";manifest_payload=ms.build_swa_final_four(checks,swa);ar.seal_file(swa);swa_descriptor=_sealed_descriptor(receipt,swa);smodel=pr.build_population_robustness_model(seed=args.seed,cell="D");smodel.load_state_dict(torch.load(swa,map_location="cpu",weights_only=False)["state_dict"],strict=True);smodel.to(device).eval();swa_state_sha=ac.state_sha256(smodel);swa_proof=_swa_runtime_proof(torch=torch,model=smodel,probe=swa_probe[0],device=device,arm_common=ac);_req(swa_proof["state_before_sha256"]==swa_proof["state_after_sha256"]==swa_state_sha,"SWA payload state identity");swa_descriptor["strict_loaded_state_sha256"]=swa_state_sha;progress["swa_published"]=True
  manifest_descriptor=_write_and_describe(receipt,out/"manifest.json",{"checkpoints":[_sealed_descriptor(receipt,p,epoch=int(p.stem[-3:])) for p in checks],"swa":swa_descriptor,"inherited_final_four":manifest_payload,"attempt":attempt_descriptor,"implementation_closure_sha256":closure["closure_sha256"]});published["swa"]=swa_descriptor;published["manifest"]=manifest_descriptor
  # The device is now legitimately occupied by this process, so the idle
  # preflight cannot be repeated.  Rebind the logical CUDA contract instead;
  # predecessor/closure are still fully revalidated at terminal time.
  final=v1.exact_source_closure(root,receipt,plan.BOUND_PATTERNS);_req(final["closure_sha256"]==closure["closure_sha256"],"closure");predfinal=profile.predecessor_validator(root);_validate_capability(capability,root=root,arm=arm,predecessor=predfinal,closure=final,gpu=capability.device,fresh=False,runtime_factory=runtime_factory,profile=profile);_req(str(v1._assert_cuda_binding_after_attempt(torch))==str(device),"final logical device drift");torch.cuda.synchronize(device)
  term={"schema":plan.SCHEMA+"_terminal","status":"PACD_FULL_TRAINING_COMPLETE","arm":arm,"arm_identity":plan.ARMS[arm],"attempt":attempt_descriptor,"launch":launch,"source_authority":source_authority,"predecessor":predfinal,"epochs":epochs,"checkpoints":[_sealed_descriptor(receipt,p,epoch=int(p.stem[-3:])) for p in checks],"swa":{**swa_descriptor,"proof":swa_proof},"manifest":manifest_descriptor,"sampler_order":order_evidence,"source_closure":{"launch":closure,"final":final},"device":{"pre":gpu,"post":gpu2,"peak_allocated":int(torch.cuda.max_memory_allocated(device)),"peak_reserved":int(torch.cuda.max_memory_reserved(device)),"tf32":{"matmul_allow_tf32":bool(torch.backends.cuda.matmul.allow_tf32),"cudnn_allow_tf32":bool(torch.backends.cudnn.allow_tf32)}},"progress":progress,"target_access":False};_write_and_describe(receipt,out/"terminal.json",term);return 0
 except BaseException as err:
  try: final=v1.exact_source_closure(root,receipt,plan.BOUND_PATTERNS)
  except BaseException as e: final={"error":str(e)}
  try: pred=profile.predecessor_validator(root)
  except BaseException as e: pred={"error":str(e)}
  v1.publish_failure(out,receipt,{"schema":plan.SCHEMA+"_failure","status":"CELL_FAILED","attempt":attempt_descriptor,"published_descriptors":published,"predecessor":pred,"source_closure":{"launch":closure,"final":final},"target_access":False,"progress":progress,"failure":{"kind":type(err).__name__,"detail":str(err),"traceback":traceback.format_exc()}});raise
