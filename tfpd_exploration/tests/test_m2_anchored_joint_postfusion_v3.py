"""No-CUDA V3 control-plane checks."""
from __future__ import annotations
import subprocess,sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def test_v3_inert_cli_and_clean_import_before_torch():
    cli=subprocess.run([sys.executable,'-S',str(ROOT/'tfpd_exploration/scripts/run_m2_anchored_joint_postfusion_v3.py')],check=True,capture_output=True,text=True)
    assert 'READY_REQUIRES_V3_OPAQUE_REVIEW' in cli.stdout
    code="import sys; import tfpd_exploration.src.m2_anchored_joint_postfusion_v3.production; print('torch' in sys.modules)"
    child=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True,env={'PYTHONPATH':str(ROOT),'PATH':'/usr/bin:/bin'})
    assert child.stdout.strip() == 'False'
def test_v3_reviewed_capability_is_opaque_and_fresh(tmp_path):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v3 import lifecycle
    reviewed=lifecycle.combined(ROOT); cap=lifecycle._issue_for_test(repo=ROOT,root=tmp_path/'v3')
    assert lifecycle.consume(cap)[0].is_dir()
def test_v3_private_synthetic_success_and_failure_prefixes(tmp_path):
    import pytest
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v3 import lifecycle,production
    reviewed=lifecycle.combined(ROOT);d=lifecycle.digest(reviewed)
    cap=lifecycle._issue_for_test(repo=ROOT,root=tmp_path/'success')
    out=production._execute_synthetic_for_test(cap=cap);root=Path(out['root']);assert (root/'terminal.json').is_file() and (root/'score'/'terminal.json').is_file()
    for label,where,prefix in [('sentinel','sentinel',['attempt.json','v2_authority.json']),('inner','inner',['attempt.json','v2_authority.json','sentinel_authority.json'])]:
        cap=lifecycle._issue_for_test(repo=ROOT,root=tmp_path/label)
        with pytest.raises(RuntimeError):production._execute_synthetic_for_test(cap=cap,fail_at=where)
        assert json.loads((tmp_path/label/'failure.json').read_text())['published_prefix']==prefix
