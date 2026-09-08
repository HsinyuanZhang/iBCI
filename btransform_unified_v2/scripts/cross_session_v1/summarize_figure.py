"""Fail-closed primary 33-cell cross-session figure generator (read-only)."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parent/"btransform_unified_v1/src")]
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
from tfpd_exploration.src.m2_dual_track_v1.plan import EXT4_SESSIONS, EXT4_EXPECTED_WINDOWS
ARMS=("Z_NONE","B_ACTIVITY_ONLY","D_JOINT")
FOLDS={"m1":{"ses-20120924","ses-20120926","ses-20120927","ses-20120928"},"m2":{"source7_ext4"},"h1":{"19250101","19250108","19250113","19250115","19250119","19250120"}}
FINAL={"m1":24,"m2":24,"h1":32}; M2_CELL="M2-CROSS-SESSION-R50-D4-ZBD-M33-V1"; LABEL={"m1":"B3S-family activity + rSyn3","m2":"HoldContrast-FiLM activity + MOVE-T4","h1":"C2-shaped activity + H-C"}
def J(p):
 try:return json.loads(p.read_text())
 except Exception as e:raise RuntimeError(f"bad JSON {p}: {e}")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x,n):
 x=float(x)
 if not math.isfinite(x):raise RuntimeError(f"nonfinite {n}")
 return x
def artifact(path,recorded,session,task):
 if not recorded or not path.is_file() or sha(path)!=recorded:raise RuntimeError(f"artifact checksum/path failure {path}")
 with np.load(path,allow_pickle=False) as z:
  if not {"target","prediction"}.issubset(z.files) or z["target"].shape!=z["prediction"].shape or not np.isfinite(z["target"]).all() or not np.isfinite(z["prediction"]).all():raise RuntimeError(f"artifact target/prediction invalid {path}")
  r=float(variance_weighted_r2(z["target"],z["prediction"])); th=hashlib.sha256(np.ascontiguousarray(z["target"]).tobytes()).hexdigest()
  required={"m1":"output_index_query_relative","m2":"eligible_starts","h1":"endpoints"}[task]
  if required not in z.files:raise RuntimeError(f"artifact lacks required coordinate {required}: {path}")
  coordinate=np.asarray(z[required]); n=len(z["target"])
  if coordinate.ndim!=1 or len(coordinate)!=n or not np.issubdtype(coordinate.dtype,np.integer) or np.any(np.diff(coordinate)<=0):raise RuntimeError(f"invalid strict coordinate {required}: {path}")
  coords=[k for k in z.files if k not in ("target","prediction")]
  h=hashlib.sha256()
  for k in sorted(coords):h.update(k.encode());h.update(np.ascontiguousarray(z[k]).tobytes())
 return r,th,h.hexdigest()
def rows(report,run,artifact_spec,task):
 ps=report.get("per_session");out={}; bind={}
 if not isinstance(ps,dict) or not ps:raise RuntimeError("per_session missing")
 for s,v in ps.items():
  score=num(v.get("r2") if isinstance(v,dict) else v,f"{run}/{s}")
  spec=artifact_spec.get(s) if artifact_spec else None
  if spec is None and isinstance(v,dict):spec=(v.get("artifact") or v.get("artifact_path"),v.get("artifact_sha256") or v.get("artifact_sha"))
  if not spec or not spec[0] or not spec[1]:raise RuntimeError(f"artifact path+recorded SHA required {run}/{s}")
  p=Path(spec[0]);p=p if p.is_absolute() else run/p
  rr,th,ch=artifact(p,spec[1],s,task)
  if abs(score-rr)>1e-8:raise RuntimeError(f"R2 artifact mismatch {run}/{s}")
  out[s]=score;bind[s]=(th,ch)
 return out,bind
def extract(cell):
 run=Path(cell.get("run") or "")
 if not run.is_dir():raise RuntimeError(f"program run path absent for {cell}")
 task,arm,fold=cell["dataset"],cell["arm"],cell["fold"]
 if arm not in ARMS or cell.get("seed")!=42 or task not in FOLDS or fold not in FOLDS[task]:raise RuntimeError(f"program identity invalid {cell}")
 train=J(run/"train_receipt.json"); scorefile=run/("target_score.json" if task=="h1" else "score_receipt.json"); score=J(scorefile)
 if task=="m1":
  meta=train
  if train.get("status")!="COMPLETED" or train.get("arm")!=arm or train.get("seed")!=42 or train.get("fold")!=fold or score.get("arm")!=arm or score.get("seed")!=42 or score.get("fold")!=fold:raise RuntimeError(f"M1 arm/seed/fold mismatch {run}")
  if score.get("target_labels_used_for_selection") is not False:raise RuntimeError(f"M1 target selection leak {run}")
  a=score.get("target_audit_arrays",{}); sel,fin=score.get("target_selected_ema"),score.get("target_epoch24_ema"); epoch=int(score.get("source_selected",{}).get("epoch",0));
  if score.get("source_selected")!=train.get("selected"):raise RuntimeError(f"M1 source-selected receipt mismatch {run}")
  specs=({fold:(run/a.get("selected",{}).get("file",""),a.get("selected",{}).get("sha256"))},{fold:(run/a.get("epoch24",{}).get("file",""),a.get("epoch24",{}).get("sha256"))})
 elif task=="m2":
  meta=J(run/"run_meta.json")
  if any(x.get("cell")!=M2_CELL for x in (meta,train,score)) or train.get("schema")!="cross_session_m2_train_receipt_v1" or train.get("status")!="COMPLETED" or any(x.get("arm")!=arm or int(x.get("seed",-1))!=42 for x in (meta,train,score)):raise RuntimeError(f"M2 identity mismatch {run}")
  if score.get("target_query_labels_used_for_selection") is not False:raise RuntimeError(f"M2 target selection leak {run}")
  sel,fin=score.get("selected_ema_ext4"),score.get("predeclared_e24_ext4_sensitivity");epoch=int(score.get("source_trial_validation_selection",{}).get("epoch",0));specs=({}, {})
  if score.get("source_trial_validation_selection")!=train.get("selection"):raise RuntimeError(f"M2 source-selected receipt mismatch {run}")
 else:
  meta=train.get("meta",{})
  if train.get("schema")!="h1_lodo_train_receipt_v2" or len(train.get("curve",[]))!=32 or len(train.get("checkpoints",{}))!=32 or meta.get("arm")!=arm or int(meta.get("seed",-1))!=42 or str(meta.get("fold")) not in (fold, f"{fold[:4]}-{fold[4:6]}-{fold[6:]}") or score.get("arm")!=arm or str(score.get("fold")) not in (fold,f"{fold[:4]}-{fold[4:6]}-{fold[6:]}"):raise RuntimeError(f"H1 identity/evidence mismatch {run}")
  if score.get("target_query_labels_used_for_selection") is not False:raise RuntimeError(f"H1 target selection leak {run}")
  sel,fin=score.get("selected_ema",{}).get("target"),score.get("fixed_e32_ema",{}).get("target");epoch=int(score.get("selected_source_epoch",0));specs=({}, {})
  if epoch!=int(train.get("selected",{}).get("epoch",0)):raise RuntimeError(f"H1 source-selected epoch mismatch {run}")
 if not isinstance(sel,dict) or not isinstance(fin,dict) or not 1<=epoch<=FINAL[task]:raise RuntimeError(f"selection/final missing {run}")
 a,b=rows(sel,run,specs[0],task);c,d=rows(fin,run,specs[1],task)
 if set(a)!=set(c):raise RuntimeError(f"endpoint sessions differ {run}")
 for report,values,name in ((sel,a,"selected"),(fin,c,"final")):
  if abs(num(report.get("equal_session_mean"),f"{run}/{name} mean")-float(np.mean(list(values.values()))))>1e-10:raise RuntimeError(f"equal_session_mean mismatch {run}/{name}")
 return {"task":task,"fold":fold,"arm":arm,"run":str(run),"selected":a,"final":c,"sb":b,"fb":d,"epoch":epoch,"receipt_sha256":sha(scorefile)}
def collect(root):
 program=J(root/"program.json");cells=program.get("cells")
 if program.get("primary_cell_count")!=33 or not isinstance(cells,list) or len(cells)!=33:raise RuntimeError("authoritative program must enumerate exactly 33 cells")
 if any(c.get("status")!="COMPLETED" for c in cells):raise RuntimeError("program has incomplete primary cells")
 data=[extract(c) for c in cells]
 if len({(r["task"],r["fold"],r["arm"]) for r in data})!=33:raise RuntimeError("duplicate/missing program cell")
 for task,want in FOLDS.items():
  got={r["fold"] for r in data if r["task"]==task}
  if got!=want:raise RuntimeError(f"{task} fold IDs mismatch: {got}")
  for fold in want:
   trio={r["arm"]:r for r in data if r["task"]==task and r["fold"]==fold}
   if set(trio)!=set(ARMS):raise RuntimeError(f"missing paired arm {task}/{fold}")
   metas=[J(Path(trio[a]["run"])/("train_receipt.json" if task=="m1" else "run_meta.json")) for a in ARMS]
   def fp(m): return {k:m.get(k) for k in ("schema","seed","fold","cell","recipe","selection","epochs","batch","context_bins","depth","attention_backend","split_contract","source_manifest_sha256")}
   if len({json.dumps(fp(m),sort_keys=True,default=str) for m in metas})!=1:raise RuntimeError(f"paired protocol metadata mismatch {task}/{fold}")
   for endpoint,bkey in (("selected","sb"),("final","fb")):
    sessions=[set(trio[x][endpoint]) for x in ARMS]
    if not all(x==sessions[0] for x in sessions[1:]):raise RuntimeError("paired session mismatch")
    for s in sessions[0]:
     if len({trio[x][bkey][s] for x in ARMS})!=1:raise RuntimeError(f"target/coordinate mismatch {task}/{fold}/{s}")
 # exact target rosters and window counts after all score receipts are validated
 for r in data:
  sessions=set(r["selected"])
  if r["task"]=="m1" and sessions!={r["fold"]}:raise RuntimeError(f"M1 target roster mismatch {r['run']}")
  if r["task"]=="m2":
   if sessions!=set(EXT4_SESSIONS):raise RuntimeError(f"M2 EXT4 roster mismatch {r['run']}")
   score=J(Path(r["run"])/"score_receipt.json")
   for session,n in EXT4_EXPECTED_WINDOWS.items():
    row=score["selected_ema_ext4"]["per_session"][session]
    if int(row.get("window_count",-1))!=n:raise RuntimeError(f"M2 window count mismatch {r['run']}/{session}")
  if r["task"]=="h1":
   iso=f"{r['fold'][:4]}-{r['fold'][4:6]}-{r['fold'][6:]}"; want=set(H1_SESSIONS_BY_DATE[iso])
   if sessions!=want:raise RuntimeError(f"H1 target roster mismatch {r['run']}")
   manifest=ROOT/"results/cross_session_v1/h1_prepared_v2"/iso/"target/manifest.json"; score=J(Path(r["run"])/"target_score.json")
   if score.get("target_manifest_sha256")!=sha(manifest):raise RuntimeError(f"H1 target manifest SHA mismatch {r['run']}")
   payload=J(manifest)
   for session in want:
    if int(score["selected_ema"]["target"]["per_session"][session].get("windows",-1))!=int(payload["records"][session]["samples"]):raise RuntimeError(f"H1 window count mismatch {r['run']}/{session}")
 return data
def summarize(data):
 out=[];means={}
 for task in FOLDS:
  for endpoint in ("selected","final"):
   foldmeans=[]
   for fold in sorted(FOLDS[task]):
    t={r["arm"]:r for r in data if r["task"]==task and r["fold"]==fold}; ss=sorted(t[ARMS[0]][endpoint]);
    for s in ss:
     z,b,d=[t[a][endpoint][s] for a in ARMS];out.append({"task":task,"fold":fold,"session":s,"endpoint":endpoint,"Z_NONE":z,"B_ACTIVITY_ONLY":b,"D_JOINT":d,"B_minus_Z":b-z,"D_minus_B":d-b})
    foldmeans.append([np.mean(list(t[a][endpoint].values())) for a in ARMS])
   means[(task,endpoint)]=np.mean(foldmeans,axis=0)
 return out,means
def plot(rows,means,out):
 """Print-oriented three-column summary; called only after all 33 cells validate."""
 fig,ax=plt.subplots(2,3,figsize=(7.15,4.45),sharey="row")
 fig.subplots_adjust(left=.075,right=.992,bottom=.235,top=.915,wspace=.24,hspace=.38)
 arm_color={"Z_NONE":"#6f6f6f","B_ACTIVITY_ONLY":"#3b75af","D_JOINT":"#c84d45"}
 title={"m1":"M1", "m2":"M2", "h1":"H1"}
 unit={"m1":"4 held-out sessions", "m2":"4 held-out sessions; 1 fit/arm", "h1":"6 held-out dates; 13 sessions"}
 for col,task in enumerate(("m1","m2","h1")):
  selected=[r for r in rows if r["task"]==task and r["endpoint"]=="selected"]
  top,bottom=ax[0,col],ax[1,col]
  for row in selected:
   top.plot(range(3),[row[arm] for arm in ARMS],color="#c7c7c7",lw=.55,zorder=1)
   bottom.plot([0,1],[row["B_minus_Z"],row["D_minus_B"]],color="#c7c7c7",lw=.55,zorder=1)
  mean=np.asarray(means[(task,"selected")],float); increments=np.asarray([mean[1]-mean[0],mean[2]-mean[1]])
  final=np.asarray(means[(task,"final")],float); final_increments=np.asarray([final[1]-final[0],final[2]-final[1]])
  top.plot(range(3),mean,color="black",lw=1.55,zorder=3)
  top.scatter(range(3),mean,s=26,c=[arm_color[a] for a in ARMS],edgecolors="black",linewidths=.35,zorder=5)
  bottom.plot([0,1],increments,color="black",lw=1.55,zorder=3)
  bottom.scatter([0,1],increments,s=26,c=[arm_color["B_ACTIVITY_ONLY"],arm_color["D_JOINT"]],edgecolors="black",linewidths=.35,zorder=5)
  bottom.plot([0,1],final_increments,color="black",lw=.9,ls=(0,(3,2)),zorder=2)
  bottom.scatter([0,1],final_increments,s=40,marker="s",facecolors="white",edgecolors="black",linewidths=.65,zorder=3.5)
  for x,value in enumerate(mean): top.annotate(f"{value:.3f}",(x,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7)
  for x,value in enumerate(increments): bottom.annotate(f"{value:+.3f}",(x,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=7)
  top.set_title(title[task],fontsize=9,fontweight="bold",pad=2);top.text(.5,.98,unit[task],transform=top.transAxes,ha="center",va="top",fontsize=6.2)
  top.set_xticks(range(3),["Z\nnone","B\nactivity","D\njoint"],fontsize=7);bottom.set_xticks([0,1],[r"$\Delta B=B-Z$",r"$\Delta$carrier$=D-B$"],fontsize=6.3)
  bottom.axhline(0,color="#4a4a4a",lw=.55,zorder=0)
  top.tick_params(axis="y",labelsize=6.5);bottom.tick_params(axis="y",labelsize=6.5)
  if col==0: top.set_ylabel(r"Target $R^2$",fontsize=7.5);bottom.set_ylabel(r"Increment in $R^2$",fontsize=7.5)
  else: top.tick_params(labelleft=False);bottom.tick_params(labelleft=False)
 fig.text(.5,.060,"Seed 42. Circles/solid: source-validation selected EMA; squares/dashed: fixed final EMA.",ha="center",fontsize=7)
 fig.text(.5,.035,"Final epoch: e24 (M1/M2), e32 (H1). Gray: paired selected sessions; black: equal-fold mean.",ha="center",fontsize=7)
 fig.text(.5,.010,"Final endpoint is a prespecified sensitivity analysis; no CI or significance test.",ha="center",fontsize=7)
 fig.savefig(out.with_suffix('.pdf'),bbox_inches="tight");fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches="tight");plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('btransform_unified_v2/results/cross_session_v1'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();data=collect(a.root);rows,means=summarize(data);a.out.mkdir(parents=True,exist_ok=False)
 with (a.out/'cross_session_figure_data.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (a.out/'cross_session_figure_summary.json').write_text(json.dumps({'seed':42,'rows':data,'means':{f'{k[0]}_{k[1]}':v.tolist() for k,v in means.items()},'H1_date_correlation':'equal six date folds; sessions within dates are not independent seeds','no_CI_pvalue_significance':True},indent=2)+'\n');plot(rows,means,a.out/'cross_session_calibration_ablation')
if __name__=='__main__':main()
