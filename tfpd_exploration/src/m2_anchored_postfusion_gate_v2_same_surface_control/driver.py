"""Opaque, inert V2 admission; physical scoring is available only post-attempt."""
from __future__ import annotations
import os, stat, time
from dataclasses import dataclass
from pathlib import Path
from . import binding, plan
class AdmissionError(RuntimeError): pass
_TOKEN=object(); _IDS=set()
@dataclass(frozen=True)
class _Capability:
    repo_root:Path; result_root:Path; parent_identity:tuple[int,int]; closure_sha256:str; _token:object; consumed:bool=False
def _env():
    got={k:os.environ.get(k) for k in plan.CPU_ENV}
    if got!=plan.CPU_ENV: raise AdmissionError(f'V2 CPU environment drift: {got}')
def _fresh(root):
    target=Path(root).absolute()/plan.RESULT_ROOT_RELATIVE; parent=target.parent; info=os.lstat(parent)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or os.path.lexists(target): raise AdmissionError('V2 result root unavailable/freshness drift')
    return target,(int(info.st_dev),int(info.st_ino))
def static_admission(root:Path): return {"schema":plan.SCHEMA,"stage":"no_data_no_cuda","public_cli_can_mint":False,"authority":plan.validate_static(root)}
def issue_live_capability(root:Path,*,token:object):
    if token is not _TOKEN: raise AdmissionError('V2 opaque issuer token mismatch')
    _env(); authority=plan.validate_static(root); target,parent=_fresh(root); cap=_Capability(Path(root).absolute(),target,parent,authority['closure_sha256'],_TOKEN); _IDS.add(id(cap)); return cap
def _mint_synthetic_capability(root): return issue_live_capability(root,token=_TOKEN)
def consume(cap):
    if not isinstance(cap,_Capability) or cap._token is not _TOKEN or id(cap) not in _IDS or cap.consumed: raise AdmissionError('V2 capability invalid/forged/reused')
    _env(); authority=plan.validate_static(cap.repo_root)
    if authority['closure_sha256']!=cap.closure_sha256: raise AdmissionError('V2 closure drift')
    target,parent=_fresh(cap.repo_root)
    if target!=cap.result_root or parent!=cap.parent_identity: raise AdmissionError('V2 root identity drift')
    object.__setattr__(cap,'consumed',True)
def execute_synthetic(cap,*,body_builder):
    """Private no-data lifecycle seam; public CLI cannot invoke it."""
    consume(cap); from . import lifecycle
    progress={'v1_predecessor_attempted':False,'v1_predecessor_verified':False,'torch_attempted':False,'target_opened':False}
    attempt={'schema':plan.SCHEMA+'_attempt_v1','status':'ATTEMPT_RESERVED','closure_sha256':cap.closure_sha256,'cpu_only':True}
    def launch(): return {'cpu_environment':dict(plan.CPU_ENV),'cuda_initialized':False,'cpu_decode_batch_size':plan.CPU_DECODE_BATCH_SIZE}
    def bodies(artifact): progress['v1_predecessor_attempted']=True; witness=binding.validate_v1_failure(cap.repo_root); progress['v1_predecessor_verified']=True; return body_builder(artifact,witness,progress)
    def revalidate():
        _env(); plan.validate_static(cap.repo_root); binding.validate_v1_failure(cap.repo_root)
        info=os.lstat(cap.result_root); _=info
    def terminal(published): return {'launch_sha256':published['launch.json'],'predecessor_authority_sha256':published['predecessor_authority.json'],'input_authority_sha256':published['input_authority.json'],'score_sha256':published['score.json'],'row_count':65,'cuda_initialized':False}
    return lifecycle.execute(cap.repo_root,attempt=attempt,launch=launch,bodies=bodies,terminal=terminal,progress=lambda:progress,revalidate=revalidate)


def _public_input_authority(materialized, prepared):
    records=materialized.get('records')
    if not isinstance(records,dict) or len(records)!=13: raise AdmissionError('V2 public input record topology drift')
    return {'schema':plan.SCHEMA+'_input_authority_v1','record_count':13,
            'records':{str(key):{str(field):value for field,value in row.items() if field!='_runtime'} for key,row in records.items()},
            'append_heldout_evidence':materialized.get('append_heldout_evidence'),
            'native_same_process_comparators':materialized.get('native_comparators'),
            'selected_evidence':prepared.get('selected_evidence'),'cuda_initialized':False}


def _authority_projection(record):
    """The V1 locked input excluding its historical CUDA comparator only."""
    return {str(key): value for key,value in record.items() if key != 'pooled_comparator'}


def _verify_v1_input_projection(*, witness, materialized):
    historical=witness['payloads']['input_authority.json']
    expected=historical.get('records'); observed=materialized.get('records')
    if not isinstance(expected,dict) or not isinstance(observed,dict) or set(expected)!=set(observed) or len(expected)!=13:
        raise AdmissionError('V2 V1-input authority topology drift')
    for key in sorted(expected):
        current={str(name):value for name,value in observed[key].items() if name != '_runtime'}
        if _authority_projection(current) != _authority_projection(expected[key]):
            raise AdmissionError(f'V2 rematerialized input authority drift: {key}')
    return {'v1_input_authority_sha256':witness['body_sha256']['input_authority.json'],
            'record_count':13,'projection_excludes':['pooled_comparator'],'historical_cuda_comparators':
            {key: expected[key]['pooled_comparator'] for key in sorted(expected)}}


def execute_production(cap):
    """Single CPU target-only V2 executor; public CLI cannot call this.

    All descriptor, checkpoint, Torch, and dataset access is nested below the
    immutable attempt (and launch) publications.  No callback can replace the
    model, causal rollout, metric, or target materializer.
    """
    consume(cap)
    from . import lifecycle, physical, laws
    progress={'stage':'reserved','rows':0,'cuda_initialized':False,'v1_predecessor_attempted':False,
              'v1_predecessor_verified':False,'pit_prepare_attempted':False,'pit_prepare_complete':False,
              'checkpoint_opened':False,'target_materialization_attempted':False,'target_materialized':False,
              'smoke_complete':False,'published_prefix':[]}
    attempt={'schema':plan.SCHEMA+'_attempt_v1','status':'ATTEMPT_RESERVED','closure_sha256':cap.closure_sha256,
             'cpu_environment':dict(plan.CPU_ENV),'cuda_initialized':False,'v1_predecessor':dict(plan.V1_BODIES)}
    def launch():
        _env(); return {'cpu_environment':dict(plan.CPU_ENV),'cuda_initialized':False,
                         'cpu_decode_batch_size':plan.CPU_DECODE_BATCH_SIZE,'execution':'single_pit_cpu_prepare'}
    def bodies(artifact):
        progress['v1_predecessor_attempted']=True; witness=binding.validate_v1_failure(cap.repo_root); progress['v1_predecessor_verified']=True
        progress['pit_prepare_attempted']=True; progress['checkpoint_opened']=True
        prepared=physical.prepare_selected_cpu_once(repo_root=cap.repo_root); progress['pit_prepare_complete']=True
        progress['target_materialization_attempted']=True
        materialized=physical.materialize_13_same_process_inputs(prepared=prepared); progress['target_materialized']=True
        predecessor=_verify_v1_input_projection(witness=witness,materialized=materialized)
        predecessor.update({'schema':plan.SCHEMA+'_predecessor_authority_v1','v1_root_relative':plan.V1_ROOT_RELATIVE,
                            'v1_root_identity':witness['root_identity'],'v1_body_sha256':witness['body_sha256']})
        predecessor_sha=artifact.publish_json('predecessor_authority.json',predecessor)
        progress['published_prefix']=['attempt.json','launch.json','predecessor_authority.json']
        input_sha=artifact.publish_json('input_authority.json',_public_input_authority(materialized,prepared))
        progress['published_prefix']=['attempt.json','launch.json','predecessor_authority.json','input_authority.json']
        records=materialized['records']; first=next(str(row['key']) for row in sorted(records.values(), key=lambda row:(plan.SURFACES.index(str(row['surface'])),str(row['session']))))
        began=time.monotonic(); smoke=physical.score_65_rows_from_materialized(prepared=prepared,materialized=materialized,record_keys=(first,)); elapsed=float(time.monotonic()-began)
        projection=elapsed*13.0; progress.update({'smoke_complete':True,'smoke_key':first,'smoke_wall_seconds':elapsed,'projected_wall_seconds':projection})
        if projection>7200.0: raise AdmissionError('V2 one-session CPU sentinel projects over 7200 seconds')
        rest=physical.score_65_rows_from_materialized(prepared=prepared,materialized=materialized,record_keys=tuple(key for key in records if key!=first))
        rows=smoke+rest; laws.validate_rows(rows); derived=laws.recompute(rows)
        score_sha=artifact.publish_json('score.json',{'schema':plan.SCHEMA+'_score_v1','rows':rows,**derived,
            'learned_refit_alpha':float(plan.REFIT_ALPHA),
            'same_process_native_zero_fixed30_exact':True,'runtime':{'cuda_initialized':False,'target_updates':0,'parameter_updates':0,
            'cpu_decode_batch_size':plan.CPU_DECODE_BATCH_SIZE,'smoke_wall_seconds':elapsed,'projected_wall_seconds':projection}})
        progress.update({'stage':'score_complete','rows':65,'published_prefix':['attempt.json','launch.json','predecessor_authority.json','input_authority.json','score.json']})
        return {'predecessor_authority.json':predecessor_sha,'input_authority.json':input_sha,'score.json':score_sha}
    def revalidate():
        _env(); static=plan.validate_static(cap.repo_root)
        if static['closure_sha256']!=cap.closure_sha256: raise AdmissionError('V2 final closure drift')
        binding.validate_v1_failure(cap.repo_root)
        info=os.lstat(cap.result_root); parent=os.lstat(cap.result_root.parent)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or (int(parent.st_dev),int(parent.st_ino))!=cap.parent_identity: raise AdmissionError('V2 reserved root drift')
    def terminal(published):
        revalidate(); return {'launch_sha256':published['launch.json'],'predecessor_authority_sha256':published['predecessor_authority.json'],'input_authority_sha256':published['input_authority.json'],
          'score_sha256':published['score.json'],'row_count':65,'cuda_initialized':False,'target_updates':0,'parameter_updates':0,
          'learned_refit_alpha':float(plan.REFIT_ALPHA),
          'v1_failure_predecessor':dict(plan.V1_BODIES),'current_closure_sha256':cap.closure_sha256,
          'limitations':['target_only_no_source_retraining','same_process_cpu_native_zero_control','no_cuda']}
    return lifecycle.execute(cap.repo_root,attempt=attempt,launch=launch,bodies=bodies,terminal=terminal,progress=lambda:dict(progress),revalidate=revalidate)
