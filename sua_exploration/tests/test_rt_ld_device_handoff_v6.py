"""Synthetic no-GPU contracts for the current CI five-arm receipt latch."""
from __future__ import annotations
import importlib.util, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; RUNNER = ROOT / "scripts/rt_ld_device_handoff_v6.py"

def _module():
    spec=importlib.util.spec_from_file_location("rt_ld_v6", RUNNER); assert spec and spec.loader
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def _write(path: Path, date: str, *, schema: str, status: str, mode: int=0o444, metrics: bool=True):
    body={"schema":schema,"status":status,"outer_date":date,"metrics":{arm:{} for arm in ("CI32-FULL","CI64-FULL","CI64-C0","CI64-LS","CI64-RS")} if metrics else {},"scope":{"formal_heldout_opened":False,"minival_opened":False,"evalai_opened":False},"deployment_updates":{"optimizer_steps":0,"backward_steps":0,"model_state_unchanged":True}}
    path.write_text(json.dumps(body)); os.chmod(path, mode)

def _paths(tmp: Path, m, partition: str): return {date: tmp/f"{date}.json" for date in m.TERMINALS[partition]}

def test_ci_receipt_missing_mode_schema_status_date_and_empty_shell_block(tmp_path: Path):
    m=_module(); paths=_paths(tmp_path,m,"gpu0_h1_static_ci64"); assert not m._ci_receipts_ready("gpu0_h1_static_ci64",paths)
    for date,path in paths.items(): _write(path,date,schema=m.CI_SCHEMA,status=m._status(date))
    assert m._ci_receipts_ready("gpu0_h1_static_ci64",paths)
    bad=next(iter(paths)); os.chmod(paths[bad],0o644); assert not m._ci_receipts_ready("gpu0_h1_static_ci64",paths)
    os.chmod(paths[bad],0o644); _write(paths[bad],bad,schema="wrong",status=m._status(bad)); assert not m._ci_receipts_ready("gpu0_h1_static_ci64",paths)
    os.chmod(paths[bad],0o644); _write(paths[bad],bad,schema=m.CI_SCHEMA,status="wrong"); assert not m._ci_receipts_ready("gpu0_h1_static_ci64",paths)
    os.chmod(paths[bad],0o644); _write(paths[bad],"19250113",schema=m.CI_SCHEMA,status=m._status("19250113")); assert not m._ci_receipts_ready("gpu0_h1_static_ci64",paths)
    os.chmod(paths[bad],0o644); _write(paths[bad],bad,schema=m.CI_SCHEMA,status=m._status(bad),metrics=False); assert not m._ci_receipts_ready("gpu0_h1_static_ci64",paths)

def test_old_phase2_receipts_even_complete_cannot_satisfy_ci_v6(tmp_path: Path):
    m=_module(); paths=_paths(tmp_path,m,"gpu1_h1_static_ci64")
    for date,path in paths.items(): _write(path,date,schema="h1_carrierid_date_lodo_phase2_terminal_evaluation_v1",status=f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_EVALUATED")
    assert not m._ci_receipts_ready("gpu1_h1_static_ci64",paths)

def test_selected_partition_ci_receipts_and_two_empty_samples_pass_without_other_partition(tmp_path: Path):
    m=_module(); paths=_paths(tmp_path,m,"gpu1_h1_static_ci64")
    for date,path in paths.items(): _write(path,date,schema=m.CI_SCHEMA,status=m._status(date))
    assert m._ci_receipts_ready("gpu1_h1_static_ci64",paths)
    assert not m._ci_receipts_ready("gpu0_h1_static_ci64",{date:tmp_path/f"missing-{date}" for date in m.TERMINALS["gpu0_h1_static_ci64"]})
    assert m._eligible_from_probes(True,("exited",False,[]),("exited",False,[]))
