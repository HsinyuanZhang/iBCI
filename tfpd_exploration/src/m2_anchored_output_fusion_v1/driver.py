from __future__ import annotations
import hashlib,os
from pathlib import Path
from . import plan
def _need(ok,msg):
 if not ok:raise RuntimeError(msg)
def execute(root:Path, *, _lifecycle=None, _gpu_attestor=None, _runner_factory=None):
 """Run the sole source-only AOF screen; private seams are test-only."""
 from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as default_imm
 imm=default_imm if _lifecycle is None else _lifecycle
 root=Path(root).absolute();_need({k:os.environ.get(k) for k in plan.ENV}==plan.ENV,'AOF environment');_need(plan.file_sha(root/plan.DESIGN_RELATIVE)==plan.DESIGN_SHA256,'AOF design SHA');_need(plan.file_sha(root/plan.WORKORDER_RELATIVE)==plan.WORKORDER_SHA256,'AOF workorder SHA');closure_sha=plan.sha(plan.jsonb(plan.closure(root)))
 class Spec:
  root_relative=plan.ROOT_RELATIVE
  def payload(self):return {'schema':plan.SCHEMA+'_root','root_relative':self.root_relative}
 a=imm.ImmutableArtifactRoot.reserve(root,Spec());attempt=a.publish_json('attempt.json',{'schema':plan.SCHEMA+'_attempt','design_sha256':plan.DESIGN_SHA256,'workorder_sha256':plan.WORKORDER_SHA256,'closure_sha256':closure_sha,'closure':plan.closure(root),'source_target_access_authorized':True,'source_target_access':False,'hidden_external_evalai_target_access':False});published={'attempt.json':attempt};progress={'stage':'attempt_published','gpu_attempted':False,'gpu_opened':False,'source_prepare_attempted':False,'source_opened':False,'source_target_access_attempted':False,'source_target_access':False,'source_targets_loader_preloaded':False,'source_target_indices_access_attempted':False,'source_target_indices_opened':False,'paired_output_attempted':False,'paired_outputs_complete':False,'hidden_external_evalai_target_access':False}
 try:
  from .runner import gpu0,StaticAOFRunner
  progress['stage']='gpu_attestation';progress['gpu_attempted']=True
  launch=(gpu0 if _gpu_attestor is None else _gpu_attestor)();progress['gpu_opened']=True;published['launch.json']=a.publish_json('launch.json',{**launch,'closure_sha256':closure_sha,'closure':plan.closure(root)})
  progress['stage']='source_prepare';progress['source_prepare_attempted']=True;progress['source_target_access_attempted']=True;progress['source_target_access']=True
  runner=(StaticAOFRunner if _runner_factory is None else _runner_factory)(root,launch);source=runner.prepare();progress['source_opened']=True;progress['source_target_access']=bool(source.get('source_target_access',True));progress['source_targets_loader_preloaded']=bool(source.get('source_targets_loader_preloaded',True));published['source_authority.json']=a.publish_json('source_authority.json',{**source,'closure_sha256':closure_sha,'parameter_updates':0,'target_parameter_updates':0})
  progress['stage']='paired_output';progress['paired_output_attempted']=True;progress['source_target_indices_access_attempted']=True
  result=runner.run();progress['source_target_indices_opened']=bool(result.get('source_target_access',False));progress['paired_outputs_complete']=True;published['paired_output_authority.json']=a.publish_json('paired_output_authority.json',{'session_evidence':result['session_evidence'],'source_split':result['source_split'],'source_targets_loader_preloaded':progress['source_targets_loader_preloaded'],'source_target_indices_opened':progress['source_target_indices_opened'],'hidden_external_evalai_target_access':False,'closure_sha256':closure_sha,'parameter_updates':0,'target_parameter_updates':0});published['fit.json']=a.publish_json('fit.json',result['fit']);published['validation.json']=a.publish_json('validation.json',result['validation']);
  if result['all7_refit'] is not None:published['all7_refit.json']=a.publish_json('all7_refit.json',result['all7_refit'])
  progress['stage']='final_revalidation';final_closure=plan.closure(root);_need(final_closure==plan.closure(root) and plan.sha(plan.jsonb(final_closure))==closure_sha,'AOF final closure drift')
  return a.publish_json('terminal.json',{'schema':plan.SCHEMA+'_terminal','status':'TERMINAL','attempt_sha256':attempt,'published':published,'source_gate':result['validation'],'all7_refit_performed':result['all7_refit'] is not None,'source_target_access':progress['source_target_access'],'source_targets_loader_preloaded':progress['source_targets_loader_preloaded'],'source_target_indices_opened':progress['source_target_indices_opened'],'hidden_external_evalai_target_access':False,'closure_sha256':closure_sha,'final_closure_sha256':plan.sha(plan.jsonb(final_closure)),'terminal_xor_failure':True}),None
 except BaseException as e:
  return None,a.publish_json('failure.json',{'schema':plan.SCHEMA+'_failure','attempt_sha256':attempt,'published_prefix':published,'progress':progress,'exception_class':f'{type(e).__module__}.{type(e).__qualname__}','message':str(e)[:300],'error_sha256':hashlib.sha256(repr(e).encode()).hexdigest(),'source_target_access':progress['source_target_access'],'source_target_indices_opened':progress['source_target_indices_opened'],'hidden_external_evalai_target_access':False,'closure_sha256':closure_sha,'terminal_xor_failure':True})
 finally:a.close()
