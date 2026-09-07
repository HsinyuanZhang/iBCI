from __future__ import annotations
import importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNNER=ROOT/"scripts/rt_ld_device_handoff_v7.py"
def _m():
 s=importlib.util.spec_from_file_location("v7",RUNNER);assert s and s.loader;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def _write(p:Path,d:str,m,arms:list[str]):
 b={"schema":m.CI_SCHEMA,"status":m.V6._status(d),"outer_date":d,"metrics":{a:{} for a in arms},"scope":{"formal_heldout_opened":False,"minival_opened":False,"evalai_opened":False},"deployment_updates":{"optimizer_steps":0,"backward_steps":0,"model_state_unchanged":True}}
 p.write_text(json.dumps(b,sort_keys=True));os.chmod(p,0o444)
def test_real_sorted_json_order_passes_and_missing_or_extra_arm_fails(tmp_path:Path):
 m=_m();name="gpu0_h1_static_ci64";paths={d:tmp_path/f"{d}.json" for d in m.TERMINALS[name]}
 for d,p in paths.items():_write(p,d,m,list(m.CI_ARMS))
 assert m._ci_receipts_ready(name,paths)
 bad=next(iter(paths));os.chmod(paths[bad],0o644);_write(paths[bad],bad,m,list(m.CI_ARMS[:-1]));assert not m._ci_receipts_ready(name,paths)
 os.chmod(paths[bad],0o644);_write(paths[bad],bad,m,list(m.CI_ARMS)+["EXTRA"]);assert not m._ci_receipts_ready(name,paths)
def test_complete_selected_partition_and_two_empty_probes_pass(tmp_path:Path):
 m=_m();name="gpu1_h1_static_ci64";paths={d:tmp_path/f"{d}.json" for d in m.TERMINALS[name]}
 for d,p in paths.items():_write(p,d,m,list(m.CI_ARMS))
 assert m._eligible_from_probes(m._ci_receipts_ready(name,paths),("exited",False,[]),("exited",False,[]))
