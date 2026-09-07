from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from sua_exploration.mc_maze import h1_event_carrier_nle5_v2 as n
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v

ROOT=Path(__file__).resolve().parents[2]
def _sessions():
 p=v.index_heldin_calib(ROOT/"SPINT-main/data/000954");return {x:v.load_event_session(p[x]) for x in v.H1_HELDIN_SESSIONS}
def test_v2_inactive_tags_absent_from_all_scoring_events():
 assert n.ACTIVE_TAGS==("Reach","Orient","Shape","Grasp","Carry","Orient2")
 for s in _sessions().values():
  for b in (3,4):assert all(e.tag in n.ACTIVE_TAGS for e in s.events_before(b)+s.events_after(b))
def test_v2_each_active_tag_has_every_outer_source_support():
 s=_sessions()
 for d in v.H1_DATES:
  events=[e for name in n.source_names_for_outer(d) for e in s[name].events]
  assert all(any(e.tag==tag for e in events) for tag in n.ACTIVE_TAGS)
def test_r1_immutable_receipt_still_verifies():
 p=ROOT/"sua_exploration/results/h1_event_carrier_nle5/source_screen_v1.json"
 out=subprocess.check_output([sys.executable,str(ROOT/"sua_exploration/scripts/verify_h1_event_carrier_nle5.py"),str(p)],text=True)
 assert json.loads(out)["status"]=="PASS_FAIL_CLOSED"
def test_v2_full_source_path():
 out=n.run_screen(_sessions())
 assert out["schema"]==n.SCHEMA and set(out["budgets"])=={"M3","M4"}
 for b in out["budgets"].values():assert len(b["nle5_sessions"])==13 and b["aggregate"]["correct_r2"]["defined_sessions"]==13
