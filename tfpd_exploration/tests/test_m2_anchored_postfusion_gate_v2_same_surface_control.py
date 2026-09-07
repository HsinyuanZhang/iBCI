"""Focused no-data/no-CUDA tests for the APFG V2 target-only repair."""
from __future__ import annotations
import json, os, shutil, stat, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
REPO=Path(__file__).resolve().parents[2]
from tfpd_exploration.src.m2_anchored_postfusion_gate_v2_same_surface_control import plan, binding, laws, driver
from tfpd_exploration.src.m2_anchored_postfusion_gate_v2_same_surface_control import physical

def _repo_with_v1(tmp_path):
    for rel in plan.CLOSURE_RELATIVES:
        dest=tmp_path/rel; dest.parent.mkdir(parents=True,exist_ok=True); os.symlink(REPO/rel,dest)
    source=REPO/plan.V1_ROOT_RELATIVE; target=tmp_path/plan.V1_ROOT_RELATIVE; target.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(source,target,copy_function=shutil.copy2)
    for leaf in target.iterdir(): leaf.chmod(0o444)
    (tmp_path/'tfpd_exploration/results').mkdir(parents=True,exist_ok=True)
    return tmp_path
def _setenv(monkeypatch):
    for k,v in plan.CPU_ENV.items(): monkeypatch.setenv(k,v)
def _row(surface,session,system,law,value):
    d='a'*64 if system!='APFG-LEARNED' else 'b'*64
    state={'NATIVE-POOLED':'e'*64,'APFG-ZERO':'f'*64,'APFG-LEARNED':'1'*64}[system]
    return {'surface':surface,'session':session,'system':system,'memory_law':law,'input_authority_key':surface+'|'+session,'prediction_sha256':d,'target_sha256':'c'*64,'query_starts_sha256':'d'*64,'window_count':10,'r2':value,'model_state_before_sha256':state,'model_state_after_sha256':state,'native_substate_sha256':plan.SELECTED_STUDENT_STATE_SHA256,'alpha_exact_positive_zero': True if system=='APFG-ZERO' else (False if system=='APFG-LEARNED' else None),'learned_refit_alpha':float(plan.REFIT_ALPHA) if system=='APFG-LEARNED' else None,'parameter_updates':0,'target_updates':0}
def _rows():
    rows=[]
    for surface,n in plan.ROSTER_SIZES.items():
      for i in range(n):
        s=f'{surface}-{i}'; rows += [_row(surface,s,'NATIVE-POOLED','FIXED30',.1),_row(surface,s,'APFG-ZERO','FIXED30',.1),_row(surface,s,'APFG-ZERO','UNCAPPED',.11),_row(surface,s,'APFG-LEARNED','FIXED30',.12),_row(surface,s,'APFG-LEARNED','UNCAPPED',.13)]
    return rows
def test_actual_v1_failure_graph_is_exact_held_fd_and_torch_free():
    witness=binding.validate_v1_failure(REPO)
    assert witness['body_sha256']==plan.V1_BODIES
    assert witness['payloads']['failure.json']['progress']['stage']=='launch'
    assert witness['payloads']['failure.json']['progress']['published_prefix'] == [
        'attempt.json','launch.json','source_authority.json','alpha_selection.json']
    assert witness['payloads']['failure.json']['published_prefix']['input_authority.json'] == plan.V1_BODIES['input_authority.json']
    # The combined V1+V2 suite imports Torch in V1 tests first.  Check the
    # intended import boundary in a clean interpreter, not via order-sensitive
    # state in this pytest worker.
    env={**os.environ,'PYTHONNOUSERSITE':'1','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONPATH':str(REPO)}
    code=("import sys; from pathlib import Path; "
          "from tfpd_exploration.src.m2_anchored_postfusion_gate_v2_same_surface_control import binding; "
          f"binding.validate_v1_failure(Path({str(REPO)!r})); "
          "assert 'torch' not in sys.modules")
    completed=subprocess.run([sys.executable,'-S','-c',code],env=env,capture_output=True,text=True,check=False)
    assert completed.returncode==0, completed.stderr
def test_v1_failure_validator_rejects_extra_leaf(tmp_path):
    root=_repo_with_v1(tmp_path); extra=root/plan.V1_ROOT_RELATIVE/'extra'; extra.write_text('x'); extra.chmod(0o444)
    with pytest.raises(binding.BindingError,match='topology'): binding.validate_v1_failure(root)

def test_v1_failure_validator_rejects_semantic_progress_normalization(tmp_path):
    root=_repo_with_v1(tmp_path); leaf=root/plan.V1_ROOT_RELATIVE/'failure.json'
    leaf.chmod(0o644)
    payload=json.loads(leaf.read_text()); payload['progress']['stage']='target'; leaf.write_text(json.dumps(payload)); leaf.chmod(0o444)
    side=leaf.with_name('failure.json.sha256'); side.chmod(0o644)
    side.write_text(__import__('hashlib').sha256(leaf.read_bytes()).hexdigest()+'  failure.json\n'); side.chmod(0o444)
    with pytest.raises(binding.BindingError,match='V1 body SHA drift'):
        binding.validate_v1_failure(root)
def test_laws_65_rows_same_surface_sentinel_and_gate():
    rows=_rows(); laws.validate_rows(rows); result=laws.recompute(rows)
    assert result['promotion_gate']['passed'] is True
    rows[1]['prediction_sha256']='f'*64
    with pytest.raises(laws.LawError,match='sentinel'): laws.validate_rows(rows)

def test_laws_rejects_learned_refit_alpha_drift():
    rows=_rows(); next(row for row in rows if row['system']=='APFG-LEARNED')['learned_refit_alpha']=0.0
    with pytest.raises(laws.LawError,match='learned refit alpha'):
        laws.validate_rows(rows)

def test_explicit_runtime_closure_covers_cpu_prepare_trace_leaves():
    # This is the repo-local subset observed in the clean source-only PIT
    # prepare/import trace.  Third-party site-packages deliberately are not
    # closure authority.
    traced={
        'tfpd_exploration/src/__init__.py',
        'tfpd_exploration/src/calibration_budget_comparators_v1.py',
        'tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py',
        'tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/plan.py',
        'tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/transition.py',
        'tfpd_exploration/src/cross_session_worst_group_v1/plan.py',
        'tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py',
        'tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py',
        'tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/physical.py',
        'tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py',
        'tfpd_exploration/src/support_anchored_t4_stage_o_v1/block_refit.py',
        'tfpd_exploration/src/support_anchored_t4_stage_o_v1/gates.py',
        'tfpd_exploration/src/support_anchored_t4_stage_o_v1/plan.py',
        'tfpd_exploration/src/support_anchored_t4_stage_o_v1/replay.py',
        'tfpd_exploration/src/support_anchored_t4_stage_p_v1/gates.py',
        'tfpd_exploration/src/support_anchored_t4_stage_p_v1/plan.py',
        'streaming_calibration_exp/src/data/afc4_xls_v2.py',
        'streaming_calibration_exp/src/data/falcon_t4_features.py',
        'streaming_calibration_exp/src/models/falcon_module.py',
        'streaming_calibration_exp/src/models/components/rt_ld_gain.py',
        'streaming_calibration_exp/src/models/components/spint.py',
        'streaming_calibration_exp/third_party/catalyst/distributed_sampler.py',
    }
    assert traced <= set(plan.CLOSURE_RELATIVES)
    assert all((REPO/rel).is_file() for rel in traced)

def test_static_closure_fails_closed_when_transitive_leaf_is_missing(tmp_path):
    root=_repo_with_v1(tmp_path)
    missing=root/'streaming_calibration_exp/src/models/components/rt_ld_gain.py'
    missing.unlink()
    with pytest.raises(ValueError,match='closure leaf missing'):
        plan.validate_static(root)
def test_inert_admission_and_actual_synthetic_lifecycle(tmp_path,monkeypatch):
    _setenv(monkeypatch); root=_repo_with_v1(tmp_path); cap=driver._mint_synthetic_capability(root)
    def build(artifact,witness,progress):
        progress['target_opened']=True
        return {'predecessor_authority.json':artifact.publish_json('predecessor_authority.json',{'v1':witness['body_sha256']}),'input_authority.json':artifact.publish_json('input_authority.json',{'records':13}),'score.json':artifact.publish_json('score.json',{'rows':65})}
    terminal,failure=driver.execute_synthetic(cap,body_builder=build)
    assert terminal and failure is None
    result=root/plan.RESULT_ROOT_RELATIVE; assert (result/'launch.json').exists() and (result/'terminal.json').exists() and not (result/'failure.json').exists()
    terminal_body=json.loads((result/'terminal.json').read_text())
    assert terminal_body['launch_sha256']==__import__('hashlib').sha256((result/'launch.json').read_bytes()).hexdigest()
    with pytest.raises(driver.AdmissionError,match='reused'): driver.execute_synthetic(cap,body_builder=build)
def test_v1_refit_alpha_literal_and_cpu_env_drift(tmp_path,monkeypatch):
    _setenv(monkeypatch); root=_repo_with_v1(tmp_path); cap=driver._mint_synthetic_capability(root); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(driver.AdmissionError,match='environment'): driver.consume(cap)
    assert np.float64(plan.REFIT_ALPHA)==np.float64(-0.20759029686450958)

def test_synthetic_failure_is_terminal_exclusive_with_attempt_launch_prefix(tmp_path,monkeypatch):
    _setenv(monkeypatch); root=_repo_with_v1(tmp_path); cap=driver._mint_synthetic_capability(root)
    def fail(_artifact,_witness,progress):
        progress['target_opened']=False
        raise RuntimeError('bounded synthetic score failure')
    terminal,failure=driver.execute_synthetic(cap,body_builder=fail)
    assert terminal is None and failure is not None
    result=root/plan.RESULT_ROOT_RELATIVE
    assert (result/'attempt.json').exists() and (result/'launch.json').exists() and (result/'failure.json').exists()
    assert not (result/'terminal.json').exists() and not (result/'score.json').exists()

def test_execute_production_typed_runtime_success_and_one_shot(tmp_path,monkeypatch):
    """Exercise the real V2 executor ordering without a loader/model callback."""
    _setenv(monkeypatch); root=_repo_with_v1(tmp_path); witness=binding.validate_v1_failure(root)
    historical=witness['payloads']['input_authority.json']['records']
    materialized={'records':{key:{**row,'pooled_comparator':{'policy':'same-process-cpu'}} for key,row in historical.items()},
                  'native_comparators':{key:{'policy':'same-process-cpu'} for key in historical},
                  'append_heldout_evidence':{'same_datamodule':True}}
    prepared={'selected_evidence':{'student_state_after_load_sha256':plan.SELECTED_STUDENT_STATE_SHA256}}
    calls=[]
    monkeypatch.setattr(physical,'prepare_selected_cpu_once',lambda **kw: calls.append('prepare') or prepared)
    monkeypatch.setattr(physical,'materialize_13_same_process_inputs',lambda **kw: calls.append('materialize') or materialized)
    all_rows=_rows()
    def score(**kw):
        keys=kw.get('record_keys'); calls.append(('score',tuple(keys) if keys is not None else None))
        return all_rows[:5] if keys is not None and len(keys)==1 else all_rows[5:]
    monkeypatch.setattr(physical,'score_65_rows_from_materialized',score)
    cap=driver._mint_synthetic_capability(root); terminal,failure=driver.execute_production(cap)
    assert terminal and failure is None and calls[0:2]==['prepare','materialize']
    result=root/plan.RESULT_ROOT_RELATIVE
    for name in ('attempt.json','launch.json','predecessor_authority.json','input_authority.json','score.json','terminal.json'):
        assert (result/name).exists() and (result/(name+'.sha256')).exists()
    terminal_body=json.loads((result/'terminal.json').read_text())
    assert terminal_body['predecessor_authority_sha256']==__import__('hashlib').sha256((result/'predecessor_authority.json').read_bytes()).hexdigest()
    with pytest.raises(driver.AdmissionError,match='reused'): driver.execute_production(cap)

def test_execute_production_failure_after_prepared_input_is_terminal_exclusive(tmp_path,monkeypatch):
    _setenv(monkeypatch); root=_repo_with_v1(tmp_path); witness=binding.validate_v1_failure(root)
    historical=witness['payloads']['input_authority.json']['records']
    prepared={'selected_evidence':{'student_state_after_load_sha256':plan.SELECTED_STUDENT_STATE_SHA256}}
    materialized={'records':{key:{**row,'pooled_comparator':{'policy':'same-process-cpu'}} for key,row in historical.items()},
                  'native_comparators':{key:{} for key in historical},'append_heldout_evidence':{}}
    monkeypatch.setattr(physical,'prepare_selected_cpu_once',lambda **kw: prepared)
    monkeypatch.setattr(physical,'materialize_13_same_process_inputs',lambda **kw: materialized)
    monkeypatch.setattr(physical,'score_65_rows_from_materialized',lambda **kw: (_ for _ in ()).throw(RuntimeError('typed score failure')))
    cap=driver._mint_synthetic_capability(root); terminal,failure=driver.execute_production(cap)
    assert terminal is None and failure is not None
    result=root/plan.RESULT_ROOT_RELATIVE
    assert (result/'predecessor_authority.json').exists() and (result/'input_authority.json').exists()
    assert (result/'failure.json').exists() and not (result/'terminal.json').exists() and not (result/'score.json').exists()
