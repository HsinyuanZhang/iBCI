"""Attempt-first APFC source-only execution; no target path exists."""
from __future__ import annotations
import hashlib, os, traceback
from pathlib import Path
from . import plan

class DriverError(RuntimeError): pass
def _need(ok,msg):
    if not ok: raise DriverError(msg)
def _env():
    got={k:os.environ.get(k) for k in plan.LIVE_ENV}; _need(got==plan.LIVE_ENV,f'APFC environment drift: {got}')
def _static(root:Path):
    _need(plan.sha256_file(root/plan.DESIGN_RELATIVE)==plan.DESIGN_SHA256,'APFC design SHA drift')
    _need(plan.sha256_file(root/plan.WORKORDER_RELATIVE)==plan.WORKORDER_SHA256,'APFC workorder SHA drift')
    files=plan.closure(root); return plan.sha256_bytes(plan.canonical_json(files))
def execute_source_screen(repo_root:Path)->tuple[str|None,str|None]:
    """Authorized source-only GPU0 execution. No target materializer is imported."""
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as immutable
    root=Path(repo_root).absolute(); _env(); closure_sha=_static(root)
    class Spec:
        root_relative=plan.RESULT_ROOT_RELATIVE
        def payload(self): return {'schema':plan.SCHEMA+'_root_v1','root_relative':self.root_relative}
    artifact=immutable.ImmutableArtifactRoot.reserve(root,Spec()); attempt=artifact.publish_json('attempt.json',{
        'schema':plan.SCHEMA+'_attempt_v1','status':'ATTEMPT_RESERVED','closure_sha256':closure_sha,
        'design_sha256':plan.DESIGN_SHA256,'workorder_sha256':plan.WORKORDER_SHA256,'target_access':False,
        'gpu_scope':'physical_gpu0_only','arms':list(plan.ARMS),'epochs':plan.EPOCHS})
    published={'attempt.json':attempt}; progress={'stage':'attempt','source_opened':False,'target_access':False,'epochs_completed':0}
    try:
        from .runner import APFCCoordinatedRunner, validate_gpu0_only
        launch=validate_gpu0_only(); published['launch.json']=artifact.publish_json('launch.json',{'schema':plan.SCHEMA+'_launch_v1',**launch,'closure_sha256':closure_sha})
        progress['stage']='source_prepare'; runner=APFCCoordinatedRunner(repo_root=root,device='cuda:0',launch_attestation=launch)
        authority=runner.prepare(); progress['source_opened']=True
        published['source_authority.json']=artifact.publish_json('source_authority.json',{'schema':plan.SCHEMA+'_source_authority_v1',**authority,'target_access':False})
        progress['stage']='screen'; result=runner.run(); progress['epochs_completed']=plan.EPOCHS
        published['screen.json']=artifact.publish_json('screen.json',{'schema':plan.SCHEMA+'_screen_v1',**result,'target_access':False})
        # Frozen fail-closed decisions.  We retain source evidence even when
        # neither capacity arm wins; that outcome is a terminal screen result.
        s=result['selected']['A-S1']; control=bool(result['scalar_control']['negative'])
        terminal={'schema':plan.SCHEMA+'_terminal_v1','status':'TERMINAL','attempt_sha256':attempt,'published':published,
                  'closure_sha256':closure_sha,'target_access':False,'source_only':True,'scalar_control_negative':control,
                  'capacity_gates':result.get('gates',{}),'selection':{a:result['selected'][a]['epoch'] for a in plan.ARMS},
                  'terminal_xor_failure':True}
        return artifact.publish_json('terminal.json',terminal),None
    except BaseException as e:
        failure={'schema':plan.SCHEMA+'_failure_v1','status':'FAILED','attempt_sha256':attempt,'published_prefix':published,
                 'progress':progress,'exception_class':f'{type(e).__module__}.{type(e).__qualname__}',
                 'message':str(e)[:300],'error_sha256':hashlib.sha256(repr(e).encode()).hexdigest(),'target_access':False,'terminal_xor_failure':True}
        return None,artifact.publish_json('failure.json',failure)
    finally: artifact.close()

__all__=('execute_source_screen','DriverError')
