"""V3 detailed CPU POOLED-sentinel authority (no training route)."""
from __future__ import annotations
import os
import subprocess,sys
from pathlib import Path
from . import lifecycle,plan
_ENV={'CUDA_VISIBLE_DEVICES':'','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
def _env(repo:Path)->dict[str,str]:
    expected={**_ENV,'PYTHONPATH':str(repo)};got={k:os.environ.get(k,'') for k in expected}
    if got!=expected:raise RuntimeError('V3 frozen CPU environment drift')
    return got
def detailed_sentinel(*,repo_root:Path,admission:dict[str,object]|None=None)->dict[str,object]:
    _env(repo_root)
    import torch
    if torch.cuda.is_initialized(): raise RuntimeError('V3 sentinel CUDA initialized')
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import production as v1,plan as p1
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical,binding
    admitted=lifecycle.admit_v2(repo_root) if admission is None else admission
    bodies=admitted['_checkpoint_bytes']
    _mods,prepared=v1._strict_cpu_modules(repo_root=repo_root,checkpoint_bytes=bodies); material=physical.materialize_13_inputs_and_pooled_comparators(prepared=prepared,pooled_comparator=None,repo_root=repo_root);held=binding.validate_pooled_comparator_score(repo_root/p1.POOLED_ROOT_RELATIVE)
    rows=[]
    for key,record in sorted(material['records'].items()):
        got=record['pooled_comparator'];want=held['comparators'].get(str(key)); fields=('prediction_sha256','target_sha256','query_starts_sha256','window_count','r2'); row={'key':str(key),'fields':{f:{'observed':got.get(f),'held':None if want is None else want.get(f),'equal':want is not None and got.get(f)==want.get(f)} for f in fields}};row['all_equal']=all(x['equal'] for x in row['fields'].values());rows.append(row)
    return {'schema':'m2_anchored_joint_postfusion_v3_sentinel_authority','cpu_only':True,'record_count':len(rows),'all_equal':all(r['all_equal'] for r in rows),'rows':rows}
def execute_score_only(*,cap:lifecycle.Capability)->dict[str,str]:
    environment=_env(cap.repo);root,root_identity=lifecycle.consume(cap);published=[]
    try:
        attempt=lifecycle.pair(root,'attempt.json',{'schema':'m2_anchored_joint_postfusion_v3_attempt','closure':cap.closure,'closure_sha256':lifecycle.digest(cap.closure),'frozen_environment':environment,'torch_imported':False,'target_access':False});published.append('attempt.json')
        admission=lifecycle.admit_v2(cap.repo);a=lifecycle.pair(root,'v2_authority.json',{k:v for k,v in admission.items() if not k.startswith('_')});published.append('v2_authority.json')
        sentinel=detailed_sentinel(repo_root=cap.repo,admission=admission)
        if sentinel['record_count']!=13 or not sentinel['all_equal']:raise RuntimeError('V3 13/13 sentinel authority failed')
        s=lifecycle.pair(root,'sentinel_authority.json',sentinel);published.append('sentinel_authority.json')
        from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1
        inner=root/'score';m=v1.closure_map(cap.repo);c=v1.issue_production_capability(repo_root=cap.repo,root=inner,reviewed=m,digest=v1.closure_sha256(m))
        # Fresh child only; no V3 process imports Torch before its attempt.
        env=dict(os.environ);env.update(environment)
        # Capability objects are intentionally not serialised; the reviewed
        # child issuer is constructed there from the same sealed V2 training root.
        code=("from pathlib import Path;from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle,production;"
              "r=Path("+repr(str(cap.repo))+");m=lifecycle.closure_map(r);c=lifecycle.issue_production_capability(repo_root=r,root=Path("+repr(str(inner))+"),reviewed=m,digest=lifecycle.closure_sha256(m));production.execute_cpu_score(cap=c,repo_root=r,training_root=r/'"+plan.V2_ROOT_RELATIVE+"/training')")
        subprocess.run([sys.executable,'-c',code],cwd=str(cap.repo),env=env,check=True)
        v1.validate_score_topology(inner,terminal=True)
        score_info=inner.stat(follow_symlinks=False)
        if inner.is_symlink(): raise RuntimeError('V3 inner score symlink')
        _score_terminal,held_score_terminal=lifecycle.held_json(inner,'terminal.json')
        final_admission=lifecycle.admit_v2(cap.repo)
        lifecycle.revalidate_cap(cap,root_identity)
        if _env(cap.repo)!=environment or lifecycle.combined(cap.repo)!=cap.closure or {k:v for k,v in final_admission.items() if not k.startswith('_')}!={k:v for k,v in admission.items() if not k.startswith('_')}:raise RuntimeError('V3 terminal predecessor/env/closure drift')
        now=inner.stat(follow_symlinks=False)
        if inner.is_symlink() or (now.st_dev,now.st_ino)!=(score_info.st_dev,score_info.st_ino) or lifecycle.held_json(inner,'terminal.json')[1]!=held_score_terminal: raise RuntimeError('V3 inner score terminal TOCTOU drift')
        t=lifecycle.pair(root,'terminal.json',{'schema':'m2_anchored_joint_postfusion_v3_terminal','attempt_sha256':attempt,'v2_authority_sha256':a,'sentinel_authority_sha256':s,'inner_score_terminal_sha256':held_score_terminal,'terminal_xor_failure':True});published.append('terminal.json');lifecycle.validate_outer(root,terminal=True);return {'root':str(root),'terminal_sha256':t}
    except Exception as e:
        if 'terminal.json' not in published:
            lifecycle.revalidate_cap(cap,root_identity);_env(cap.repo);lifecycle.admit_v2(cap.repo)
            lifecycle.pair(root,'failure.json',{'schema':'m2_anchored_joint_postfusion_v3_failure','published_prefix':published,'exception_class':type(e).__name__,'terminal_xor_failure':False});lifecycle.validate_outer(root,terminal=False,published=tuple(published))
        raise

def _execute_synthetic_for_test(*,cap:lifecycle.Capability,fail_at:str|None=None)->dict[str,str]:
    """Private typed outer-route harness; never imports Torch or opens data."""
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1
    root,identity=lifecycle.consume(cap);published=[]
    try:
        attempt=lifecycle.pair(root,'attempt.json',{'schema':'m2_anchored_joint_postfusion_v3_attempt','closure':cap.closure,'synthetic':True});published.append('attempt.json')
        authority=lifecycle.admit_v2(cap.repo);a=lifecycle.pair(root,'v2_authority.json',{k:v for k,v in authority.items() if not k.startswith('_')});published.append('v2_authority.json')
        if fail_at=='sentinel':raise RuntimeError('synthetic sentinel failure')
        sentinel={'schema':'m2_anchored_joint_postfusion_v3_sentinel_authority','cpu_only':True,'record_count':13,'all_equal':True,'rows':[{'key':str(i),'all_equal':True} for i in range(13)]}
        s=lifecycle.pair(root,'sentinel_authority.json',sentinel);published.append('sentinel_authority.json')
        score=root/'score';score.mkdir()
        if fail_at=='inner':raise RuntimeError('synthetic inner failure')
        for name in v1.SCORE_BODIES:
            v1._pair(score,name,{'schema':'synthetic-v1-score','terminal_xor_failure':True} if name=='terminal.json' else {'name':name})
        v1.validate_score_topology(score,terminal=True)
        t=lifecycle.pair(root,'terminal.json',{'schema':'m2_anchored_joint_postfusion_v3_terminal','attempt_sha256':attempt,'v2_authority_sha256':a,'sentinel_authority_sha256':s,'inner_score_terminal_sha256':lifecycle.sha(score/'terminal.json'),'terminal_xor_failure':True});published.append('terminal.json');lifecycle.validate_outer(root,terminal=True);return {'root':str(root),'terminal_sha256':t}
    except Exception as e:
        if 'terminal.json' not in published:
            lifecycle.pair(root,'failure.json',{'schema':'m2_anchored_joint_postfusion_v3_failure','published_prefix':published,'exception_class':type(e).__name__,'terminal_xor_failure':False});lifecycle.validate_outer(root,terminal=False,published=tuple(published))
        raise
