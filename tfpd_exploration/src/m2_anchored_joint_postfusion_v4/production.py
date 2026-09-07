"""V4 route: one CPU materialization bridges historical GPU POOLED to CPU AJPF."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
from . import lifecycle
_ENV={'CUDA_VISIBLE_DEVICES':'','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
def _env(repo:Path)->dict[str,str]:
    want={**_ENV,'PYTHONPATH':str(repo)};got={k:os.environ.get(k,'') for k in want}
    if got!=want:raise RuntimeError('V4 frozen CPU environment drift')
    return got
def _bridge(*,repo:Path,admission:Mapping[str,Any])->tuple[dict[str,Any],Any,Mapping[str,Any],dict[str,Any]]:
    """The only strict-load/preparation/materialization; returns objects for scoring."""
    _env(repo); import torch
    if torch.cuda.is_initialized(): raise RuntimeError('V4 CUDA initialized')
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production as v1
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    modules,prepared=v1._strict_cpu_modules(repo_root=repo,checkpoint_bytes=dict(admission['_checkpoint_bytes']))
    # Native CPU generation is deliberately retained for the bridge; it is not
    # the practical comparator used by the final gates.
    material=physical.materialize_13_inputs_and_pooled_comparators(prepared=prepared,pooled_comparator=None,repo_root=repo)
    held=admission['_pooled']['comparators']; rows=[]; practical={}
    for key,record in sorted(material['records'].items()):
        observed=record['pooled_comparator']; sealed=held.get(str(key))
        if not isinstance(sealed,Mapping):raise RuntimeError('V4 held POOLED key absent')
        fields={name:{'observed':observed.get(name),'held':sealed.get(name),'equal':observed.get(name)==sealed.get(name)} for name in ('prediction_sha256','target_sha256','query_starts_sha256','window_count','r2')}
        r2_delta=abs(float(observed['r2'])-float(sealed['r2']))
        rows.append({'key':str(key),'fields':fields,'r2_absolute_delta':r2_delta})
        practical[str(record['session'])]=sealed
    exact_target=sum(bool(r['fields']['target_sha256']['equal']) for r in rows);exact_starts=sum(bool(r['fields']['query_starts_sha256']['equal']) for r in rows);exact_windows=sum(bool(r['fields']['window_count']['equal']) for r in rows);exact_predictions=sum(bool(r['fields']['prediction_sha256']['equal']) for r in rows)
    passed=(len(rows)==13 and exact_target==13 and exact_starts==13 and exact_windows==13
            and max(r['r2_absolute_delta'] for r in rows)<=2e-7)
    bridge={'schema':'m2_anchored_joint_postfusion_v4_cross_device_bridge','cpu_only':True,'materialization_count':1,'record_count':len(rows),'rows':rows,
            'target_sha256_equal_count':exact_target,'query_starts_sha256_equal_count':exact_starts,'window_count_equal_count':exact_windows,'prediction_sha256_equal_count':exact_predictions,
            'max_r2_absolute_delta':max(r['r2_absolute_delta'] for r in rows),'r2_tolerance':2e-7,
            'prediction_equality_is_not_required':True,'practical_comparator':'sealed_historical_gpu_pooled_r2',
            'passed':passed}
    return bridge,torch,modules,{'materialized':material,'practical':practical}
def _input_authority(materialized:Mapping[str,Any])->dict[str,Any]:
    records=materialized['records'];return {'schema':'m2_anchored_joint_postfusion_v4_input_authority','record_count':len(records),'keys':sorted(records),
            'target_sha256':{k:str(v['target_sha256']) for k,v in sorted(records.items())},'query_starts_sha256':{k:str(v['query_starts_sha256']) for k,v in sorted(records.items())}}
def execute_score_only(*,cap:lifecycle.Capability)->dict[str,str]:
    environment=_env(cap.repo);root,identity=lifecycle.consume(cap);published=[];admission=None
    try:
        attempt=lifecycle.pair(root,'attempt.json',{'schema':'m2_anchored_joint_postfusion_v4_attempt','closure':cap.closure,'closure_sha256':lifecycle.digest(cap.closure),'frozen_environment':environment,'torch_imported':False,'target_access':False});published.append('attempt.json')
        admission=lifecycle.admit_predecessors(cap.repo);pred=lifecycle.pair(root,'predecessor_authority.json',{k:v for k,v in admission.items() if not k.startswith('_')});published.append('predecessor_authority.json')
        bridge,torch,modules,bundle=_bridge(repo=cap.repo,admission=admission);bridge_sha=lifecycle.pair(root,'cross_device_bridge.json',bridge);published.append('cross_device_bridge.json')
        if not bridge['passed']:raise RuntimeError('V4 cross-device bridge authority failed')
        inp=lifecycle.pair(root,'input_authority.json',_input_authority(bundle['materialized']));published.append('input_authority.json')
        from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import scorer
        rows=scorer.score_whole_stack_78_cpu(torch=torch,modules=modules,materialized=bundle['materialized']);score=lifecycle.pair(root,'score.json',{'schema':'m2_anchored_joint_postfusion_v4_score','row_count':len(rows),'rows':rows});published.append('score.json')
        ext=[r['session'] for r in bundle['materialized']['records'].values() if r['surface']=='external_post30_local'];within=[r['session'] for r in bundle['materialized']['records'].values() if r['surface']=='within_post30']
        gates=scorer.build_preregistered_gates(rows=rows,historical_pooled=bundle['practical'],external_sessions=tuple(sorted(ext)),within_sessions=tuple(sorted(within)));gate=lifecycle.pair(root,'gates.json',gates);published.append('gates.json')
        lifecycle.revalidate(cap,identity,admission);_env(cap.repo)
        terminal=lifecycle.pair(root,'terminal.json',{'schema':'m2_anchored_joint_postfusion_v4_terminal','attempt_sha256':attempt,'predecessor_authority_sha256':pred,'cross_device_bridge_sha256':bridge_sha,'input_authority_sha256':inp,'score_sha256':score,'gates_sha256':gate,'terminal_xor_failure':True});published.append('terminal.json');lifecycle.validate_outer(root,terminal=True);return {'root':str(root),'terminal_sha256':terminal}
    except Exception as error:
        if 'terminal.json' not in published:
            if admission is not None:lifecycle.revalidate(cap,identity,admission)
            else:
                # Attempt remains truthful for pre-admission faults; closure/root/env
                # are still repeated before a failure leaf.
                p=cap.root.parent.stat(follow_symlinks=False);r=cap.root.stat(follow_symlinks=False)
                if ((p.st_dev,p.st_ino)!=cap.parent_identity or (r.st_dev,r.st_ino)!=identity
                        or lifecycle.combined(cap.repo)!=cap.closure):
                    raise RuntimeError('V4 pre-admission boundary drift')
            _env(cap.repo)
            lifecycle.pair(root,'failure.json',{'schema':'m2_anchored_joint_postfusion_v4_failure','published_prefix':published,'exception_class':type(error).__name__,'terminal_xor_failure':False});lifecycle.validate_outer(root,terminal=False,published=tuple(published))
        raise
def _execute_synthetic_for_test(*,cap:lifecycle.Capability,fail_at:str|None=None)->dict[str,str]:
    """Private typed publication harness: no data/Torch and no public callbacks."""
    root,identity=lifecycle.consume(cap);published=[]
    try:
        a=lifecycle.pair(root,'attempt.json',{'schema':'m2_anchored_joint_postfusion_v4_attempt','synthetic':True});published.append('attempt.json')
        p=lifecycle.pair(root,'predecessor_authority.json',{'schema':'synthetic'});published.append('predecessor_authority.json')
        b=lifecycle.pair(root,'cross_device_bridge.json',{'schema':'m2_anchored_joint_postfusion_v4_cross_device_bridge','synthetic':True,'passed':fail_at!='bridge'});published.append('cross_device_bridge.json')
        if fail_at=='bridge':raise RuntimeError('synthetic bridge')
        if fail_at=='score':raise RuntimeError('synthetic score')
        i=lifecycle.pair(root,'input_authority.json',{'schema':'synthetic'});published.append('input_authority.json');s=lifecycle.pair(root,'score.json',{'schema':'synthetic'});published.append('score.json');g=lifecycle.pair(root,'gates.json',{'schema':'synthetic'});published.append('gates.json')
        t=lifecycle.pair(root,'terminal.json',{'schema':'m2_anchored_joint_postfusion_v4_terminal','attempt_sha256':a,'predecessor_authority_sha256':p,'cross_device_bridge_sha256':b,'input_authority_sha256':i,'score_sha256':s,'gates_sha256':g,'terminal_xor_failure':True});published.append('terminal.json');lifecycle.validate_outer(root,terminal=True);return {'root':str(root),'terminal_sha256':t}
    except Exception as error:
        lifecycle.pair(root,'failure.json',{'schema':'m2_anchored_joint_postfusion_v4_failure','published_prefix':published,'exception_class':type(error).__name__,'terminal_xor_failure':False});lifecycle.validate_outer(root,terminal=False,published=tuple(published));raise
