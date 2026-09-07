"""CPU/no-data checks for the V4 score-only outer controller."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
from tfpd_exploration.src.m2_anchored_joint_postfusion_v4 import lifecycle, production

def _cap(tmp_path:Path):
    parent=tmp_path/'parent';parent.mkdir()
    return lifecycle._issue_for_test(repo_root=ROOT,outer_root=parent/'v4')

def test_v4_private_synthetic_success_consumes_opaque_cap_and_seals_exact_outer(tmp_path:Path):
    result=production._execute_synthetic_for_test(cap=_cap(tmp_path))
    root=Path(result['root'])
    lifecycle.validate_outer(root,terminal=True)
    terminal,d=lifecycle.held_json(root,'terminal.json')
    assert d==result['terminal_sha256']
    assert terminal['terminal_xor_failure'] is True
    assert set(item.name for item in root.iterdir())==set(lifecycle.OUTER_PREFIX)|{'terminal.json'}|{x+'.sha256' for x in (*lifecycle.OUTER_PREFIX,'terminal.json')}

def test_v4_terminal_rejects_resealed_body_link_drift(tmp_path:Path):
    result=production._execute_synthetic_for_test(cap=_cap(tmp_path))
    root=Path(result['root']); terminal_path=root/'terminal.json'; side=root/'terminal.json.sha256'
    terminal=json.loads(terminal_path.read_text());terminal['score_sha256']='0'*64
    raw=(json.dumps(terminal,sort_keys=True,indent=2)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
    terminal_path.chmod(0o644);side.chmod(0o644);terminal_path.write_bytes(raw);side.write_text(f'{digest}  terminal.json\n');terminal_path.chmod(0o444);side.chmod(0o444)
    with pytest.raises(lifecycle.Error):lifecycle.validate_outer(root,terminal=True)

@pytest.mark.parametrize(('stage','prefix'),[('bridge',('attempt.json','predecessor_authority.json','cross_device_bridge.json')),('score',('attempt.json','predecessor_authority.json','cross_device_bridge.json'))])
def test_v4_private_synthetic_failures_keep_exact_immutable_prefix(tmp_path:Path,stage:str,prefix:tuple[str,...]):
    cap=_cap(tmp_path)
    with pytest.raises(RuntimeError): production._execute_synthetic_for_test(cap=cap,fail_at=stage)
    root=cap.root;lifecycle.validate_outer(root,terminal=False,published=prefix)
    failure,d=lifecycle.held_json(root,'failure.json')
    assert tuple(failure['published_prefix'])==prefix and len(d)==64
    assert not (root/'terminal.json').exists()
    if stage=='bridge':
        bridge,_=lifecycle.held_json(root,'cross_device_bridge.json')
        assert bridge['passed'] is False

def test_v4_capability_requires_external_exact_combined_map_and_canonical_root(tmp_path:Path):
    reviewed=lifecycle.combined(ROOT); parent=ROOT/'tfpd_exploration/results'
    # Test only the rejected arbitrary-root path; never reserve canonical V4 root.
    with pytest.raises(lifecycle.Error):
        lifecycle.issue_after_independent_review(reviewed_map=reviewed,reviewed_digest=lifecycle.digest(reviewed),repo_root=ROOT,outer_root=tmp_path/'arbitrary')
    bad=dict(reviewed);bad['v4/x']='0'*64
    with pytest.raises(lifecycle.Error):
        lifecycle.issue_after_independent_review(reviewed_map=bad,reviewed_digest=lifecycle.digest(bad),repo_root=ROOT,outer_root=ROOT/'tfpd_exploration/results/m2_anchored_joint_postfusion_v4')

def test_v4_held_json_rejects_resealed_sidecar_mode_and_link(tmp_path:Path):
    root=tmp_path/'held';root.mkdir();raw=b'{"x": 1}\n';d=hashlib.sha256(raw).hexdigest()
    (root/'x.json').write_bytes(raw);(root/'x.json.sha256').write_text(f'{d}  x.json\n')
    (root/'x.json').chmod(0o444);(root/'x.json.sha256').chmod(0o644)
    with pytest.raises(lifecycle.Error):lifecycle.held_json(root,'x.json')

def test_v4_held_file_rejects_nonimmutable_incident_shape(tmp_path:Path):
    leaf=tmp_path/'incident.md';leaf.write_text('x');leaf.chmod(0o644)
    with pytest.raises(lifecycle.Error): lifecycle.held_file(leaf)

def test_v4_admits_exact_held_v3_v2_and_pooled_authorities_without_torch():
    admission=lifecycle.admit_predecessors(ROOT)
    assert admission['v3_failure_sha256']=='a1ee808841e699842d361a889d9945d35340efad97c11ca87d97dd3a4f2b69b1'
    assert admission['v2_training']['v2_training_terminal_sha256']=='d0e913e7a4f765f8c011556cb843753bf4734e290d80125af74c06cf051df4c3'
    assert admission['pooled_score_sha256']=='455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6'
    assert admission['historical_pooled_device']['batch_size']==1024
    assert admission['historical_pooled_device']['cuda_visible_devices']=='1'

def test_v4_inert_cli_is_import_safe_and_never_opens_torch(tmp_path:Path):
    script=ROOT/'tfpd_exploration/scripts/run_m2_anchored_joint_postfusion_v4.py'
    namespace={};exec(compile(script.read_text(),str(script),'exec'),namespace)
    assert callable(namespace['main'])

def test_v4_clean_process_can_import_route_before_torch_or_target_access():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONPATH':str(ROOT)}
    code="import sys; from tfpd_exploration.src.m2_anchored_joint_postfusion_v4 import production; assert 'torch' not in sys.modules; print('clean')"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    assert result.stdout.strip()=='clean'

def test_v4_clean_qualified_import_trace_is_contained_by_combined_closure():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONPATH':str(ROOT)}
    code="""import json,sys
from pathlib import Path
from tfpd_exploration.src.m2_anchored_joint_postfusion_v4 import lifecycle,production
from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import scorer
from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical,binding
r=Path('/home/xinyuan/Work_host/SPINT').resolve(); c=lifecycle.combined(r)
def norm(k):
 i=k.find('tfpd_exploration/'); return k[i:] if i>=0 else k
allowed={norm(k) for k in c}; loaded=set()
for m in tuple(sys.modules.values()):
 p=getattr(m,'__file__',None)
 if p:
  try:
   q=str(Path(p).resolve().relative_to(r))
   if q.endswith('.py'): loaded.add(q)
  except ValueError: pass
print(json.dumps({'missing':sorted(loaded-allowed),'torch':('torch' in sys.modules)}))"""
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    traced=json.loads(result.stdout)
    assert traced=={'missing':[],'torch':False}
