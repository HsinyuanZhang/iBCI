#!/usr/bin/env python3
"""Verifier for the immutable post-r1 ontology-corrected NLE5 v2 receipt."""
from __future__ import annotations
import argparse,hashlib,json,math,stat
from pathlib import Path
import numpy as np
SCHEMA="h1_event_carrier_nle5_source_screen_v2";HSE="e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f";TAGS=("Reach","Orient","Shape","Grasp","Carry","Orient2")
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4<<20),b""):h.update(b)
 return h.hexdigest()
def summ(rows):
 x=np.asarray([z for _,z in rows]);i=int(np.argmax(abs(x)));return {"defined_sessions":13,"mean":float(x.mean()),"median":float(np.median(x)),"positive":int((x>0).sum()),"zero":int((x==0).sum()),"negative":int((x<0).sum()),"leave_largest_absolute_out_mean":float(np.delete(x,i).mean()),"removed_session":rows[i][0]}
def chk(a,b):
 for k in ("defined_sessions","positive","zero","negative","removed_session"):need(a[k]==b[k],"summary "+k)
 for k in ("mean","median","leave_largest_absolute_out_mean"):need(math.isclose(a[k],b[k],rel_tol=0,abs_tol=1e-12),"summary "+k)
def pos(x):return x["defined_sessions"]==13 and x["mean"]>0 and x["median"]>0 and x["positive"]>=10 and x["leave_largest_absolute_out_mean"]>0
def verify(path):
 p=Path(path).resolve();side=p.with_suffix(p.suffix+".sha256");need(p.is_file() and side.is_file() and stat.S_IMODE(p.stat().st_mode)==stat.S_IMODE(side.stat().st_mode)==0o444,"immutable");digest=sha(p);need(side.read_text().strip()==f"{digest}  {p.name}","sha");x=json.loads(p.read_text());need(x["schema"]==SCHEMA and x["ontology_binding"]["hse5_receipt_sha256"]==HSE and tuple(x["ontology_binding"]["active_tags"])==TAGS and x["ontology_binding"]["r1_ontology_correction"] is True,"ontology binding")
 pre=Path(x["predeclaration_path"]);ps=pre.with_suffix(pre.suffix+".sha256");need(pre.is_file() and ps.is_file() and stat.S_IMODE(pre.stat().st_mode)==stat.S_IMODE(ps.stat().st_mode)==0o444 and sha(pre)==x["predeclaration_sha256"],"predeclare");q=json.loads(pre.read_text());need(q["post_r1_ontology_correction"] is True and q["bound_hse5_receipt_sha256"]==HSE and tuple(q["active_tags"])==TAGS,"pre ontology")
 need(x["baseline_reproduction"]["passed"] and x["baseline_reproduction"]["comparisons"]==78 and x["baseline_reproduction"]["maximum_absolute_difference"]<=1e-10,"HSE reproduce")
 for k in ("module","runner","event_parser","hse5"):need(sha(x["implementation_binding"][k+"_path"])==x["implementation_binding"][k+"_sha256"],"binding "+k)
 names=tuple(x["source_binding"]["sessions"]);need(len(names)==13 and len(set(names))==13,"source count")
 for f in x["source_binding"]["files"]:need(sha(f["path"])==f["sha256"] and not any(z in f["path"].lower() for z in ("minival","heldout","formal","evalai")),"input hash/scope")
 reports={};passed=[]
 for b in (3,4):
  q=x["budgets"][f"M{b}"];rows=q["nle5_sessions"];base=q["hse5_baseline_sessions"];need(tuple(rows)==tuple(base)==names,"rows")
  for outer_date,m in q["nonlinear_map_by_outer_date"].items():need(tuple(m["active_tags"])==TAGS and m["raw_width"]==15 and all(not s.startswith("ses-"+outer_date) for s in m["source_sessions"]),"map separation")
  for r in rows.values():need(r["carrier_dim"]==5 and r["target_design_rank"]==5 and r["outer_future_used_only_for_scoring"] and r["rotation_invariance_max_abs_error"]<=1e-11 and all(r[z]["fixed_points"]==0 for z in ("label_shuffle","tag_shuffle","row_shuffle")),"target/control")
  fields={"correct_r2":[(n,rows[n]["median_r2_correct"]) for n in names],"correct_minus_hse5":[(n,rows[n]["median_r2_correct"]-base[n]["median_r2_correct"]) for n in names],"correct_minus_label_shuffle":[(n,rows[n]["median_delta_label_shuffle"]) for n in names],"correct_minus_tag_shuffle":[(n,rows[n]["median_delta_tag_shuffle"]) for n in names],"correct_minus_row_shuffle":[(n,rows[n]["median_delta_row_shuffle"]) for n in names],"correct_minus_intercept":[(n,rows[n]["median_delta_intercept"]) for n in names]}
  for k,z in fields.items():chk(q["aggregate"][k],summ(z))
  a=q["aggregate"];material=pos(a["correct_minus_hse5"]) and a["correct_minus_hse5"]["mean"]>=.02 and a["correct_minus_hse5"]["median"]>=.01;controls={k:pos(a[k]) for k in ("correct_minus_label_shuffle","correct_minus_tag_shuffle","correct_minus_row_shuffle","correct_minus_intercept")};expected=bool(material and all(controls.values()));need(q["gate"]["passed"]==expected,"gate");reports[f"M{b}"]=a
  if expected:passed.append(f"M{b}")
 status="PASS_CPU_NLE5_V2_MATERIAL" if len(passed)==2 else "STOP_CPU_NLE5_V2_NOT_MATERIAL";need(x["status"]==status and not x["gpu_authorized_by_this_screen"],"status")
 return {"status":"PASS","receipt":str(p),"receipt_sha256":digest,"terminal_status":status,"passing_budgets":passed,"budget_report":reports}
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("receipt");print(json.dumps(verify(a.parse_args().receipt),indent=2,sort_keys=True))
