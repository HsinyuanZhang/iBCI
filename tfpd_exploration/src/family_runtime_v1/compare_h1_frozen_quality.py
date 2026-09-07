"""Archive-only descriptive comparison of frozen H1 references (no model imports)."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from .complete_h1_family_source import artifact_audit, atomic_json, metric, require_same_files, sha, validate_archive

COUNT, SESSIONS = 20325, 13
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = ROOT / "results/family_runtime_v1/original_h1_frozen_same20325_v1"
ORIGINAL_RECEIPT_SHA = "539bfa832c650df51e22be14b5ee0dba0c1f49fbb595f9fb869328a601f30c48"
ORIGINAL_NPZ_SHA = "f1bb6739415ea66bb86bf4285035f131333e350072ae364139b81488912ba81c"
C2_PATH = ROOT / "results/decoder_validation_v2/20260905_190000/h1/c2_epoch15_same_surface_v1/c2_epoch15_complete_stream.json"
C2_SHA = "96a9a496141e2bae533facb009903417f19a78ea970b0d3dc11c8742835d5c98"

def _finite_metrics(value):
    names=("r2_concat_float64","equal_session_mean_r2_float64","worst_session_r2_float64")
    if any(name not in value or not np.isfinite(value[name]) for name in names): raise RuntimeError("nonfinite metric receipt")
    per=value.get("per_session_r2_float64",{})
    if len(per)!=SESSIONS or not all(np.isfinite(x) for x in per.values()): raise RuntimeError("13 finite per-session metrics required")

def _same_metrics(actual, reported, *, prefix=""):
    _finite_metrics(actual); _finite_metrics(reported)
    for name in ("r2_concat_float64","equal_session_mean_r2_float64","worst_session_r2_float64"):
        if abs(actual[name]-reported[name])>1e-10: raise RuntimeError(prefix+"reported metric drift: "+name)
    if actual["per_session_r2_float64"].keys()!=reported["per_session_r2_float64"].keys(): raise RuntimeError(prefix+"per-session keys drift")
    for key,value in actual["per_session_r2_float64"].items():
        if abs(value-reported["per_session_r2_float64"][key])>1e-10: raise RuntimeError(prefix+"per-session metric drift")

def _original_metric(receipt, arrays):
    if (receipt.get("status")!="PASS_AS_SHIPPED_ORIGINAL_H1_REFERENCE_ONLY" or receipt.get("scored_count")!=COUNT
        or receipt.get("public_calls")!=20920 or receipt.get("pre")!=receipt.get("post")
        or receipt.get("direct_native_count")!=65 or receipt.get("max_native_abs_error")!=0):
        raise RuntimeError("original frozen receipt authority/native/count drift")
    if sha(receipt["_path"])!=ORIGINAL_RECEIPT_SHA: raise RuntimeError("original receipt SHA drift")
    if sha(receipt["_npz_path"])!=ORIGINAL_NPZ_SHA: raise RuntimeError("original archive SHA drift")
    if receipt.get("archive",{}).get("sha256")!=ORIGINAL_NPZ_SHA: raise RuntimeError("original receipt archive SHA drift")
    rows=receipt.get("sessions",[])
    if len(rows)!=SESSIONS or any(row.get("original_calibration_shape")!=[2,1024,176] for row in rows): raise RuntimeError("original calibration/session receipt drift")
    actual=metric(arrays); _same_metrics(actual,receipt.get("metrics",{}),prefix="original "); return actual

def _c2(c2, authority, sessions):
    if sha(C2_PATH)!=C2_SHA or c2.get("status")!="COMPLETE_FIXED_REFERENCE_SCORE" or c2.get("mode")!="complete" or c2.get("n_bins")!=COUNT:
        raise RuntimeError("C2 historical receipt identity/count drift")
    if c2.get("input_authority")!=authority: raise RuntimeError("C2 source-cache authority/arrays drift")
    reported={"r2_concat_float64":c2.get("r2_concat"),"equal_session_mean_r2_float64":c2.get("equal_session_mean_r2"),"worst_session_r2_float64":c2.get("worst_session_r2"),"per_session_r2_float64":c2.get("per_session_r2")}
    _finite_metrics(reported)
    if set(reported["per_session_r2_float64"])!=set(sessions): raise RuntimeError("C2/original session roster drift")
    if abs(reported["equal_session_mean_r2_float64"]-float(np.mean(list(reported["per_session_r2_float64"].values()))))>1e-7 or abs(reported["worst_session_r2_float64"]-min(reported["per_session_r2_float64"].values()))>1e-7: raise RuntimeError("C2 imported aggregate/per-session drift")
    if "readout" not in c2.get("selection_disclosure","").lower(): raise RuntimeError("C2 fixed-readout disclosure missing")
    return reported

def _deltas(name, candidate, original):
    per={s: candidate["per_session_r2_float64"][s]-original["per_session_r2_float64"][s] for s in original["per_session_r2_float64"]}
    return {"candidate":name,"pooled_delta":candidate["r2_concat_float64"]-original["r2_concat_float64"],"equal_session_mean_delta":candidate["equal_session_mean_r2_float64"]-original["equal_session_mean_r2_float64"],"worst_session_delta":candidate["worst_session_r2_float64"]-original["worst_session_r2_float64"],"per_session_delta":per,"win_sessions":sum(v>0 for v in per.values()),"tie_sessions":sum(v==0 for v in per.values()),"loss_sessions":sum(v<0 for v in per.values())}

def same_surface(candidate, original):
    for key in ("target","end","session_id"):
        if not np.array_equal(candidate[key],original[key]): raise RuntimeError("same-20325 target/end/session identity drift: "+key)

def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if os.environ.get("H1_FROZEN_QUALITY_COMPARISON_GO")!="1": raise RuntimeError("explicit archive-only comparison GO required")
    audit=artifact_audit(args.formal)
    # Hash fixed bytes before receipt parsing or NPZ deserialization.
    if sha(args.original_receipt)!=ORIGINAL_RECEIPT_SHA or sha(args.original_npz)!=ORIGINAL_NPZ_SHA or sha(C2_PATH)!=C2_SHA: raise RuntimeError("fixed original/C2 receipt or archive SHA drift")
    receipt=json.loads(args.original_receipt.read_text()); receipt["_path"]=str(args.original_receipt); receipt["_npz_path"]=str(args.original_npz)
    original_input=args.original_receipt.parent/"input_authority.json"
    if not original_input.is_file() or json.loads(original_input.read_text()).get("pre")!=receipt.get("pre"): raise RuntimeError("original receipt/input authority pre-image drift")
    authority_path=Path(receipt["pre"]["authority"]["path"])
    if sha(authority_path)!=receipt["pre"]["authority"]["sha256"]: raise RuntimeError("original authority file SHA drift")
    formal_input=args.formal/"input_authority.json"
    immutable={str(path):sha(path) for path in (Path(__file__),Path(__file__).with_name("complete_h1_family_source.py"),Path(__file__).with_name("h1_replay_contract.py"),args.original_receipt,args.original_npz,C2_PATH,authority_path,formal_input,original_input)}
    # All inputs below are NPZ/JSON only; no code_source_audit/cache/model imports.
    original_arrays=validate_archive(args.original_npz); original=_original_metric(receipt,original_arrays)
    authority=json.loads(authority_path.read_text())
    formal_binding=json.loads(formal_input.read_text()).get("bindings",{})
    if (formal_binding.get("source_cache_sha256")!=receipt["pre"]["cache"]["sha256"]
            or formal_binding.get("source_authority_sha256")!=receipt["pre"]["authority"]["sha256"]):
        raise RuntimeError("formal/original source-cache authority metadata drift")
    c2=_c2(json.loads(C2_PATH.read_text()),authority,original["per_session_r2_float64"])
    tables={"ORIGINAL_frozen":original}; arrays={"ORIGINAL_frozen":original_arrays}
    for arm in ("flat","route"):
        for label in ("selected","epoch12"):
            path=args.formal/"exports"/f"{arm}_{label}_complete_native_float64.npz"; value=validate_archive(path)
            same_surface(value,original_arrays)
            observed=metric(value); reported=audit["final"]["complete"][arm]["reports"][label]["complete"]; _same_metrics(observed,reported,prefix=f"{arm}/{label} ")
            key=arm.upper()+"_"+label; tables[key]=observed; arrays[key]=value
    # C2 deliberately has no prediction NPZ, so its historical aggregate is imported, never recomputed.
    tables["C2_historical_aggregate_imported"]={**c2,"independently_recomputed":False}
    post=artifact_audit(args.formal)
    if post!=audit: raise RuntimeError("formal authority changed during archive comparison")
    require_same_files(audit["files"]); require_same_files(immutable)
    family={k:v for k,v in tables.items() if k.startswith(("FLAT_","ROUTE_"))}
    pair={f"{a}_minus_{b}":_deltas(a,family[a],family[b]) for a in family for b in family if a<b}
    result={"schema":"h1_frozen_same20325_descriptive_quality_table_v1","status":"PASS_DESCRIPTIVE_ARCHIVE_COMPARISON_ONLY","pre_artifact":audit,"post_artifact":post,"tables":tables,"paired_deltas_vs_original":{k:_deltas(k,v,original) for k,v in tables.items() if k not in ("ORIGINAL_frozen",)},"family_pair_deltas":pair,"delta_definition":"difference_of_worst_session_scores compares each candidate minimum; minima may be different sessions","original_calibration":"as-shipped actual calibration [2,1024,176]","family_calibration":"explicit frozen M3 banks","c2_readout":"historical fixed readout identity, NOT FITTED by this comparison","c2_prediction_npz_available":False,"cache_disclosure":"source cache is not reread; its immutable byte SHA and authority metadata are bound from receipts","comparability":"descriptive only; not a controlled training, support-budget, selection-exposure, or calibration-method ablation","noninferiority":"no prospective NI threshold; no formal NI declaration","immutable_pre":immutable,"immutable_post":require_same_files(immutable)}
    atomic_json(result,args.output); return result

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--formal",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--original-receipt",type=Path,default=ORIGINAL_DIR/"receipt.json"); p.add_argument("--original-npz",type=Path,default=ORIGINAL_DIR/"original_h1_minival_native_float64.npz")
 result=run(p.parse_args(argv)); print(json.dumps({k:result[k] for k in ("status","schema")})); return result
if __name__=="__main__": main()
