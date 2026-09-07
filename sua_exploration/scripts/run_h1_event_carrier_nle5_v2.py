#!/usr/bin/env python3
"""Write NLE5 v2's ontology-corrected immutable source screen."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
import argparse,hashlib,json,stat,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
from sua_exploration.mc_maze import h1_event_carrier_nle5_v2 as n
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as h
HSE=ROOT/"sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json";OUT=ROOT/"sua_exploration/results/h1_event_carrier_nle5";DATA=ROOT/"SPINT-main/data/000954"
HSE_SHA="e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f"
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):
    q=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):q.update(b)
    return q.hexdigest()
def write(path,x):
    path=Path(path).resolve();side=path.with_suffix(path.suffix+".sha256");need(not path.exists() and not side.exists(),f"no overwrite {path}");path.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=path.parent,prefix="."+path.name,suffix=".tmp");tmp=Path(t)
    try:
        with os.fdopen(fd,"wb") as f:f.write((json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n").encode());f.flush();os.fsync(f.fileno())
        os.chmod(tmp,0o444);os.link(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()
    d=sha(path);fd=os.open(side,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,"w") as f:f.write(f"{d}  {path.name}\n");f.flush();os.fsync(f.fileno())
    return path,d
def tag_counts(ref):
    # The V2 receipt gives each session's M3 support plus later; sum all M3/M4-independent event tags safely from source parser after binding receipt.
    return {tag:sum(row["support_events_by_tag"].get(tag,0) for row in ref["budgets"]["M3"]["sessions"].values()) for tag in v.MOVEMENT_TAGS}
def pre(ref_sha,counts):return {"schema":"h1_event_carrier_nle5_predeclaration_v2","protocol":n.PROTOCOL,"written_before_nwb_open":True,"post_r1_ontology_correction":True,"r1_preserved_immutable":True,"bound_hse5_receipt_path":str(HSE.resolve()),"bound_hse5_receipt_sha256":ref_sha,"active_tags":list(n.ACTIVE_TAGS),"inactive_tags":["Release","SnapTo"],"hse5_receipt_global_tag_counts":counts,"frozen_model":{"raw_width":15,"rff_width":16,"rff_seed":190271,"source_ridge_lambda":1.,"target_ridge_lambda":3.,"carrier_dim":5},"no_grid_expansion":True,"controls":["endpoint_label_shuffle","tag_shuffle","orthogonal_coordinate_rotation_invariance","row_attachment_shuffle","intercept_only"],"material_gate":{"mean":.02,"median":.01,"positive":"10/13","leave_one_out_positive":True},"prohibitions":["GPU","formal","minival","heldout","dense_velocity"]}
def reproduce(body,ref):
    ds=[]
    for b in (3,4):
      for s in v.H1_HELDIN_SESSIONS:
       for a,z in (("median_r2_correct","median_r2_correct"),("median_delta_label_shuffle","median_delta_shuffle"),("median_delta_intercept","median_delta_intercept")):ds.append(abs(body["budgets"][f"M{b}"]["hse5_baseline_sessions"][s][a]-ref["budgets"][f"M{b}"]["sessions"][s]["forward"][z]))
    need(max(ds)<=1e-10,"HSE exact");return {"passed":True,"comparisons":78,"absolute_tolerance":1e-10,"maximum_absolute_difference":max(ds),"reference_path":str(HSE.resolve()),"reference_sha256":sha(HSE)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,default=DATA);p.add_argument("--output-dir",type=Path,default=OUT);a=p.parse_args();os.nice(19)
 ref=json.loads(HSE.read_text());need(sha(HSE)==HSE_SHA and ref["schema"]==h.SCHEMA,"bound HSE receipt SHA/schema");counts=tag_counts(ref);need(tuple(k for k,x in counts.items() if x>0)==n.ACTIVE_TAGS and all(counts[x]==0 for x in n.INACTIVE_TAGS),"HSE active ontology mismatch")
 prepath=a.output_dir/"predeclaration_v2.json";prefile,presh=write(prepath,pre(HSE_SHA,counts));paths=v.index_heldin_calib(a.data_root);sessions={x:v.load_event_session(paths[x]) for x in v.H1_HELDIN_SESSIONS};need(all(e.tag in n.ACTIVE_TAGS for s in sessions.values() for e in s.events),"parsed inactive tag")
 started=time.monotonic();body=n.run_screen(sessions);body.update({"runtime_seconds":time.monotonic()-started,"predeclaration_path":str(prefile),"predeclaration_sha256":presh,"ontology_binding":{"hse5_receipt_path":str(HSE.resolve()),"hse5_receipt_sha256":HSE_SHA,"hse5_global_tag_counts":counts,"active_tags":list(n.ACTIVE_TAGS),"r1_ontology_correction":True},"baseline_reproduction":reproduce(body,ref),"source_binding":{"sessions":list(v.H1_HELDIN_SESSIONS),"dates":list(v.H1_DATES),"files":[{"session":x,"path":str(paths[x]),"sha256":sessions[x].input_sha256} for x in v.H1_HELDIN_SESSIONS]},"implementation_binding":{"module_path":str(Path(n.__file__).resolve()),"module_sha256":sha(n.__file__),"runner_path":str(Path(__file__).resolve()),"runner_sha256":sha(__file__),"event_parser_path":str(Path(v.__file__).resolve()),"event_parser_sha256":sha(v.__file__),"hse5_path":str(Path(h.__file__).resolve()),"hse5_sha256":sha(h.__file__)},"receipt_integrity":{"thread_limits":{k:os.environ[k] for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")},"nice_requested":True}});out,d=write(a.output_dir/"source_screen_v2.json",body);print(json.dumps({"status":body["status"],"receipt":str(out),"sha256":d,"M3":body["budgets"]["M3"]["aggregate"],"M4":body["budgets"]["M4"]["aggregate"]},indent=2,sort_keys=True));return 0 if body["status"]=="PASS_CPU_NLE5_V2_MATERIAL" else 2
if __name__=="__main__":raise SystemExit(main())
